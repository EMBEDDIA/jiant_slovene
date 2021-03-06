""" Trainer """
import os
import re
import math
import glob
import time
import copy
import random
import logging as log
import itertools
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_
from tensorboardX import SummaryWriter  # pylint: disable=import-error

from allennlp.common import Params  # pylint: disable=import-error
from allennlp.common.checks import ConfigurationError  # pylint: disable=import-error
from allennlp.data.iterators import BasicIterator, BucketIterator  # pylint: disable=import-error
from allennlp.training.learning_rate_schedulers import LearningRateScheduler  # pylint: disable=import-error
from allennlp.training.optimizers import Optimizer  # pylint: disable=import-error
from allennlp.nn.util import move_to_device, device_mapping

from .utils.utils import assert_for_log  # pylint: disable=import-error
from .evaluate import evaluate
from .utils import config


def build_trainer_params(args, task_names):
    ''' In an act of not great code design, we wrote this helper function which
    extracts trainer parameters from args. In particular, we want to search args
    for task specific training parameters. '''
    def _get_task_attr(attr_name): return config.get_task_attr(args, task_names,
                                                               attr_name)
    params = {}
    train_opts = ['optimizer', 'lr', 'batch_size', 'lr_decay_factor',
                  'lr_patience', 'patience', 'scheduler_threshold']
    # we want to pass to the build_train()
    extra_opts = ['sent_enc', 'd_hid', 'warmup',
                  'max_grad_norm', 'min_lr', 'batch_size',
                  'cuda', 'keep_all_checkpoints',
                  'val_data_limit', 'max_epochs_per_task', 'training_data_fraction']
    for attr in train_opts:
        params[attr] = _get_task_attr(attr)
    for attr in extra_opts:
        params[attr] = getattr(args, attr)
    params['max_vals'] = _get_task_attr('max_vals')
    params['val_interval'] = _get_task_attr('val_interval')
    params['dec_val_scale'] = _get_task_attr('dec_val_scale')

    return Params(params)


def build_trainer(params, model, run_dir, metric_should_decrease=True):
    '''Build a trainer from params.

    Parameters
    ----------
    params: Trainer parameters as built by build_trainer_params.
    model: A module with trainable parameters.
    run_dir: The directory where we save the models.

    Returns
    -------
    A trainer object, a trainer config object, an optimizer config object,
        and a scheduler config object.
    '''

    opt_params = {'type': params['optimizer'], 'lr': params['lr']}
    if params['optimizer'] == 'adam':
        # AMSGrad is a flag variant of Adam, not its own object.
        opt_params['amsgrad'] = True
    elif params['optimizer'] == 'bert_adam':
        opt_params['t_total'] = -1
        opt_params['warmup'] = 0.1
    opt_params = Params(opt_params)

    if 'transformer' in params['sent_enc']:
        assert False, "Transformer is not yet tested, still in experimental stage :-("
        schd_params = Params({'type': 'noam',
                              'model_size': params['d_hid'],
                              'warmup_steps': params['warmup'],
                              'factor': 1.0})
        log.info('\tUsing noam scheduler with warmup %d!', params['warmup'])
    else:
        schd_params = Params({'type': 'reduce_on_plateau',
                              'mode': 'min' if metric_should_decrease else 'max',
                              'factor': params['lr_decay_factor'],
                              'patience': params['lr_patience'],
                              'threshold': params['scheduler_threshold'],
                              'threshold_mode': 'abs',
                              'verbose': True})
        log.info('\tUsing ReduceLROnPlateau scheduler!')

    train_params = Params({'cuda_device': params['cuda'],
                           'patience': params['patience'],
                           'grad_norm': params['max_grad_norm'],
                           'val_interval': params['val_interval'],
                           'max_vals': params['max_vals'],
                           'lr_decay': .99, 'min_lr': params['min_lr'],
                           'keep_all_checkpoints': params['keep_all_checkpoints'],
                           'val_data_limit': params['val_data_limit'],
                           'max_epochs_per_task': params['max_epochs_per_task'],
                           'dec_val_scale': params['dec_val_scale'],
                           'training_data_fraction': params['training_data_fraction']})
    trainer = SamplingMultiTaskTrainer.from_params(model, run_dir,
                                                   copy.deepcopy(train_params))
    return trainer, train_params, opt_params, schd_params


class SamplingMultiTaskTrainer:
    def __init__(self, model, patience=2, val_interval=100, max_vals=50,
                 serialization_dir=None, cuda_device=-1,
                 grad_norm=None, grad_clipping=None, lr_decay=None, min_lr=None,
                 keep_all_checkpoints=False, val_data_limit=5000, max_epochs_per_task=-1,
                 dec_val_scale=100, training_data_fraction=1.0):
        """
        The training coordinator. Unusually complicated to handle MTL with tasks of
        diverse sizes.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        patience , optional (default=2)
            Number of epochs to be patient before early stopping.
        val_metric , optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        serialization_dir , optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device , optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        keep_all_checkpoints : If set, keep checkpoints from every validation. Otherwise, keep only
            best and (if different) most recent.
        val_data_limit: During training, use only the first N examples from the validation set.
            Set to -1 to use all.
        training_data_fraction: If set to a float between 0 and 1, load only the specified percentage
            of examples. Hashing is used to ensure that the same examples are loaded each epoch.
        """
        self._model = model

        self._patience = patience
        self._max_vals = max_vals
        self._val_interval = val_interval
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._lr_decay = lr_decay
        self._min_lr = min_lr
        self._keep_all_checkpoints = keep_all_checkpoints
        self._val_data_limit = val_data_limit
        self._max_epochs_per_task = max_epochs_per_task
        self._dec_val_scale = dec_val_scale
        self._training_data_fraction = training_data_fraction
        self._task_infos = None
        self._metric_infos = None

        self._log_interval = 10  # seconds
        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

        self._TB_dir = None
        if self._serialization_dir is not None:
            self._TB_dir = os.path.join(self._serialization_dir, "tensorboard")
            self._TB_train_log = SummaryWriter(
                os.path.join(self._TB_dir, "train"))
            self._TB_validation_log = SummaryWriter(
                os.path.join(self._TB_dir, "val"))

    def _check_history(self, metric_history, cur_score, should_decrease=False):
        '''
        Given a the history of the performance on a metric
        and the current score, check if current score is
        best so far and if out of patience.
        '''
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        if should_decrease:
            index_of_last_improvement = metric_history.index(min(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (patience + 1)
        else:
            index_of_last_improvement = metric_history.index(max(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (patience + 1)

        return best_so_far, out_of_patience

    def _setup_training(
            self,
            tasks,
            batch_size,
            train_params,
            optimizer_params,
            scheduler_params,
            phase):
        ''' Set up the trainer by initializing task_infos and metric_infos, which
        track necessary information about the training status of each task and metric respectively.

        Returns:
            - task_infos (Dict[str:Dict[str:???]]): dictionary containing where each task_info contains:
                - iterator: a task specific (because it uses that task's fields to dynamically batch) batcher
                - n_tr_batches: the number of training batches
                - tr_generator: generator object that returns the batches, set to repeat indefinitely
                - loss: the accumulated loss (during training or validation)
                - n_batches_since_val: number of batches trained on since the last validation
                - total_batches_trained: number of batches trained over all validation checks
                - optimizer: a task specific optimizer, not used if the global optimizer is not None
                - scheduler: a task specific scheduler, not used if the global optimizer is not None
                - stopped: a bool indicating if that task is stopped or not (if it ran out of patience or hit min lr)
                - last_log: the time we last logged progress for the task

            - metric_infos (Dict[str:Dict[str:???]]): dictionary containing metric information.
                Each metric should be the validation metric of a task, except {micro/macro}_avg,
                which are privileged to get an aggregate multi-task score. Each dict contains:
                - hist (List[float]): previous values of the metric
                - stopped (Bool): whether or not that metric is stopped or not
                - best (Tuple(Int, Dict)): information on the best value of that metric and when it happened
        '''
        task_infos = {task.name: {} for task in tasks}
        for task in tasks:
            task_info = task_infos[task.name]

            # Adding task-specific smart iterator to speed up training
            instance = [i for i in itertools.islice(task.train_data, 1)][0]
            pad_dict = instance.get_padding_lengths()
            sorting_keys = []
            for field in pad_dict:
                for pad_field in pad_dict[field]:
                    sorting_keys.append((field, pad_field))
            iterator = BucketIterator(sorting_keys=sorting_keys,
                                      max_instances_in_memory=10000,
                                      batch_size=batch_size,
                                      biggest_batch_first=True)
            tr_generator = iterator(task.train_data, num_epochs=None)
            task_info['iterator'] = iterator

            if phase == "main":
                # Warning: This won't be precise when training_data_fraction is set, since each example is included
                # or excluded independently using a hashing function. Fortunately, it
                # doesn't need to be.
                task_info['n_tr_batches'] = math.ceil(
                    task.n_train_examples * self._training_data_fraction / batch_size)
            else:
                task_info['n_tr_batches'] = math.ceil(task.n_train_examples / batch_size)

            task_info['tr_generator'] = tr_generator
            task_info['loss'] = 0.0
            task_info['total_batches_trained'] = 0
            task_info['n_batches_since_val'] = 0
            task_info['optimizer'] = Optimizer.from_params(train_params,
                                                           copy.deepcopy(optimizer_params))
            task_info['scheduler'] = LearningRateScheduler.from_params(
                task_info['optimizer'], copy.deepcopy(scheduler_params))
            task_info['stopped'] = False
            task_info['last_log'] = time.time()

        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks] + ['micro_avg', 'macro_avg']
        metric_infos = {metric: {'hist': [], 'stopped': False, 'best': (-1, {})} for
                        metric in all_metrics}
        self._task_infos = task_infos
        self._metric_infos = metric_infos
        return task_infos, metric_infos

    def train(self, tasks, stop_metric,
              batch_size, n_batches_per_pass,
              weighting_method, scaling_method,
              train_params, optimizer_params, scheduler_params,
              shared_optimizer=1, load_model=1, phase="main"):
        """
        The main training loop.
        Training will stop if we run out of patience or hit the minimum learning rate.

        Parameters
        ----------
        tasks: A list of task objects to train on.
        stop_metric: The metric to use for early stopping.
        batch_size: The batch size to use for the tasks
        n_batches_per_pass: How many training steps per task per pass.
        weighting_method: How to sample which task to use.
        scaling_method: How to scale gradients.
        train_params: Trainer config object.
        optimizer_params: Optimizer config object.
        scheduler_params: Scheduler config object.
        shared_optimizer: Use a single optimizer object for all tasks in MTL. Recommended.
        load_model: Whether to restore and continue training if a checkpoint is found.
        phase: Usually 'main' or 'eval'.

        Returns
        -------
        Validation results
        """
        validation_interval = self._val_interval
        task_infos, metric_infos = self._setup_training(tasks, batch_size, train_params,
                                                        optimizer_params, scheduler_params, phase)

        if shared_optimizer:  # if shared_optimizer, ignore task_specific optimizers
            if self._max_epochs_per_task:
                n_epoch_steps = sum([info["n_tr_batches"] for info in task_infos.values()])
                optimizer_params["t_total"] = n_epoch_steps * self._max_epochs_per_task
            g_optimizer = Optimizer.from_params(train_params, copy.deepcopy(optimizer_params))
            g_scheduler = LearningRateScheduler.from_params(
                g_optimizer, copy.deepcopy(scheduler_params))
        else:
            g_optimizer, g_scheduler = None, None
        self._g_optimizer = g_optimizer
        self._g_scheduler = g_scheduler

        n_pass, should_stop = 0, False  # define these here b/c they might get overridden on load
        if self._serialization_dir is not None and phase != "eval":  # Resume from serialization path
            if load_model and any(
                    ["model_state_" in x for x in os.listdir(self._serialization_dir)]):
                n_pass, should_stop = self._restore_checkpoint()
                log.info("Loaded model from checkpoint. Starting at pass %d.", n_pass)
            else:
                log.info("Not loading.")
                checkpoint_pattern = os.path.join(
                    self._serialization_dir, "*_{}_*.th".format(phase))
                assert_for_log(len(glob.glob(checkpoint_pattern)) == 0,
                               "There are existing checkpoints in %s which will be overwritten. "
                               "Use load_model = 1 to load the checkpoints instead. "
                               "If you don't want them, delete them or change your experiment name." %
                               self._serialization_dir)

        if self._grad_clipping is not None:  # pylint: disable=invalid-unary-operand-type
            def clip_function(grad): return grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        # Calculate per task sampling weights
        assert_for_log(len(tasks) > 0, "Error: Expected to sample from 0 tasks.")

        task_names = [task.name for task in tasks]
        task_n_train_examples = np.array([task.n_train_examples for task in tasks])
        task_n_train_batches = np.array([task_infos[task.name]['n_tr_batches'] for task in tasks])
        log.info("Training examples per task: " + str(dict(zip(task_names, task_n_train_examples))))

        if weighting_method == 'uniform':
            sample_weights = [1.0] * len(tasks)
            log.info("Sampling tasks uniformly.")
        elif weighting_method == 'proportional':
            sample_weights = task_n_train_examples.astype(float)
            log.info("Sampling tasks proportional to number of training examples.")
        elif weighting_method == 'proportional_log_batch':
            sample_weights = np.log(task_n_train_batches)
            log.info("Sampling tasks proportional to log number of training batches.")
        elif weighting_method == 'proportional_log_example':
            sample_weights = np.log(task_n_train_examples)
            log.info("Sampling tasks proportional to log number of training examples.")
        elif weighting_method == 'inverse':
            sample_weights = 1 / task_n_train_examples
            log.info("Sampling tasks inverse to number of training examples.")
        elif weighting_method == 'inverse_log_example':
            sample_weights = 1 / np.log(task_n_train_examples)
            log.info("Sampling tasks inverse to log number of training examples.")
        elif weighting_method == 'inverse_log_batch':
            sample_weights = 1 / np.log(task_n_train_batches)
            log.info("Sampling tasks inverse to log number of training batches.")
        elif 'power_' in weighting_method:
            weighting_power = float(weighting_method.strip('power_'))
            sample_weights = task_n_train_examples ** weighting_power
            log.info("Sampling tasks with %s.", weighting_method.replace('_', ' of '))
        elif 'softmax_' in weighting_method:  # exp(x/temp)
            weighting_temp = float(weighting_method.strip('softmax_'))
            sample_weights = np.exp(task_n_train_examples / weighting_temp)
            log.info("Sampling tasks with %s.", weighting_method.replace('_', ' of temperature '))

        normalized_sample_weights = np.array(sample_weights) / sum(sample_weights)
        log.info("Using weighting method: %s, with normalized sample weights %s ",
                 weighting_method, np.array_str(normalized_sample_weights, precision=4))

        # Sample the tasks to train on. Do it all at once (val_interval) for MAX EFFICIENCY.
        samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

        if scaling_method == 'uniform':
            scaling_weights = [1.0] * len(tasks)
        elif scaling_method == 'max_proportional':
            scaling_weights = task_n_train_examples.astype(float)
        elif scaling_method == 'max_proportional_log':
            scaling_weights = np.log(task_n_train_examples)
        elif 'max_power_' in scaling_method:
            scaling_power = float(scaling_method.strip('max_power_'))
            scaling_weights = task_n_train_examples ** scaling_power
        elif scaling_method == 'max_inverse_log':
            scaling_weights = 1 / np.log(task_n_train_examples)
        elif scaling_method == 'max_inverse':
            scaling_weights = 1 / task_n_train_examples
        # Weighting losses based on best epochs for each task from a previous uniform run, normalizd by max epoch
        # eg. 'max_epoch_9_18_1_11_18_2_14_16_1'
        elif 'max_epoch_' in scaling_method:
            epochs = scaling_method.strip('max_epoch_').split('_')
            assert len(epochs) == len(tasks), "Loss Scaling Error: epoch number not match."
            scaling_weights = np.array(list(map(int, epochs)))

        # normalized by max weight
        if 'max' in scaling_method:
            scaling_weights = scaling_weights / np.max(scaling_weights)

        scaling_weights = dict(zip(task_names, scaling_weights))
        log.info(
            "Using loss scaling method: %s, with weights %s",
            scaling_method,
            str(scaling_weights))

        offset = 0
        all_tr_metrics = {}
        log.info("Beginning training. Stopping metric: %s", stop_metric)
        while not should_stop:
            self._model.train()
            task = samples[(n_pass + offset) % validation_interval]  # randomly select a task
            task_info = task_infos[task.name]
            if task_info['stopped']:
                offset += 1
                continue
            tr_generator = task_info['tr_generator']
            optimizer = g_optimizer if shared_optimizer else task_info['optimizer']
            scheduler = g_scheduler if shared_optimizer else task_info['scheduler']
            total_batches_trained = task_info['total_batches_trained']
            n_batches_since_val = task_info['n_batches_since_val']
            tr_loss = task_info['loss']
            for batch in itertools.islice(tr_generator, n_batches_per_pass):
                n_batches_since_val += 1
                total_batches_trained += 1
                optimizer.zero_grad()
                output_dict = self._forward(batch, task=task, for_training=True)
                assert_for_log("loss" in output_dict,
                               "Model must return a dict containing a 'loss' key")
                loss = output_dict["loss"]  # optionally scale loss

                loss *= scaling_weights[task.name]

                loss.backward()
                #assert_for_log(not torch.isnan(loss).any(), "NaNs in loss.")
                tr_loss += loss.data.cpu().numpy()

                # Gradient regularization and application
                if self._grad_norm:
                    clip_grad_norm_(self._model.parameters(), self._grad_norm)
                optimizer.step()
                n_pass += 1  # update per batch

                # step scheduler if it's not ReduceLROnPlateau
                if not isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                    scheduler.step_batch(n_pass)

            # Update training progress on that task
            task_info['n_batches_since_val'] = n_batches_since_val
            task_info['total_batches_trained'] = total_batches_trained
            task_info['loss'] = tr_loss

            # Intermediate log to logger and tensorboard
            if time.time() - task_info['last_log'] > self._log_interval:
                task_metrics = task.get_metrics()

                # log to tensorboard
                if self._TB_dir is not None:
                    task_metrics_to_TB = task_metrics.copy()
                    task_metrics_to_TB["loss"] = \
                        float(task_info['loss'] / n_batches_since_val)
                    self._metrics_to_tensorboard_tr(n_pass, task_metrics_to_TB, task.name)

                task_metrics["%s_loss" % task.name] = tr_loss / n_batches_since_val
                description = self._description_from_metrics(task_metrics)
                log.info("Update %d: task %s, batch %d (%d): %s", n_pass,
                         task.name, n_batches_since_val, total_batches_trained, description)
                task_info['last_log'] = time.time()

                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric()
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

            # Validation
            if n_pass % validation_interval == 0:

                # Dump and log all of our current info
                epoch = int(n_pass / validation_interval)
                log.info("***** Pass %d / Epoch %d *****", n_pass, epoch)
                # Get metrics for all training progress so far
                for task in tasks:
                    task_info = task_infos[task.name]
                    n_batches_since_val = task_info['n_batches_since_val']
                    if n_batches_since_val > 0:
                        task_metrics = task.get_metrics(reset=True)
                        for name, value in task_metrics.items():
                            all_tr_metrics["%s_%s" % (task.name, name)] = value
                        all_tr_metrics["%s_loss" % task.name] = \
                            float(task_info['loss'] / n_batches_since_val)
                    else:
                        all_tr_metrics["%s_loss" % task.name] = 0.0
                    log.info("%s: trained on %d batches, %.3f epochs", task.name,
                             n_batches_since_val, n_batches_since_val / task_info['n_tr_batches'])
                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric(reset=True)
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

                # Validate
                log.info("Validating...")
                all_val_metrics, should_save, new_best_macro = self._validate(
                    epoch, tasks, batch_size, periodic_save=(phase != "eval"))

                # Check stopping conditions
                should_stop = self._check_stop(epoch, stop_metric, tasks)

                # Log results to logger and tensorboard
                for name, value in all_val_metrics.items():
                    log.info("Statistic: %s", name)
                    if name in all_tr_metrics:
                        log.info("\ttraining: %3f", all_tr_metrics[name])
                    log.info("\tvalidation: %3f", value)
                if self._TB_dir is not None:
                    self._metrics_to_tensorboard_val(n_pass, all_val_metrics)
                lrs = self._get_lr()  # log LR
                for name, value in lrs.items():
                    log.info("%s: %.6f", name, value)
                elmo_params = self._model.get_elmo_mixing_weights(tasks)
                if elmo_params:  # log ELMo mixing weights
                    for task_name, task_params in elmo_params.items():
                        log.info("ELMo mixing weights for {}:".format(task_name))
                        log.info("\t" + ", ".join(["{}: {:.6f}".format(layer, float(param))
                                                   for layer, param in task_params.items()]))

                # Reset training preogress
                all_tr_metrics = {}
                samples = random.choices(
                    tasks,
                    weights=sample_weights,
                    k=validation_interval)  # pylint: disable=no-member

                if should_save:
                    self._save_checkpoint(
                        {"pass": n_pass, "epoch": epoch, "should_stop": should_stop},
                        phase=phase, new_best_macro=new_best_macro)

        log.info('Stopped training after %d validation checks', n_pass / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)  # , validation_interval)

    def _aggregate_results(self, tasks, task_infos, metric_infos):
        ''' Helper function to print results after finishing training '''
        results = {}
        for task in tasks:
            task_info = task_infos[task.name]
            log.info('Trained %s for %d batches or %.3f epochs',
                     task.name, task_info['total_batches_trained'],
                     task_info['total_batches_trained'] / task_info['n_tr_batches'])
            results[task.name] = metric_infos[task.val_metric]['best'][0]  # * validation_interval
        results['micro'] = metric_infos['micro_avg']['best'][0]  # * validation_interval
        results['macro'] = metric_infos['macro_avg']['best'][0]  # * validation_interval
        log.info('***** VALIDATION RESULTS *****')
        for metric in metric_infos.keys():
            best_epoch, epoch_metrics = metric_infos[metric]['best']
            all_metrics_str = ', '.join(['%s: %.5f' % (metric, score) for
                                         metric, score in epoch_metrics.items()])
            log.info('%s, %d, %s', metric, best_epoch, all_metrics_str)
        return results

    def _validate(self, epoch, tasks, batch_size, periodic_save=True):
        ''' Validate on all tasks and return the results and whether to save this epoch or not '''
        task_infos, metric_infos = self._task_infos, self._metric_infos
        g_scheduler = self._g_scheduler
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        all_val_metrics["macro_avg"] = 0.0
        all_val_metrics["micro_avg"] = 0.0
        n_examples_overall = 0.0

        # Get validation numbers for each task
        for task in tasks:
            n_examples, batch_num = 0, 0
            task_info = task_infos[task.name]

            # to speed up training, we evaluate on a subset of validation data
            if self._val_data_limit >= 0:
                max_data_points = min(task.n_val_examples, self._val_data_limit)
            else:
                max_data_points = task.n_val_examples
            val_generator = BasicIterator(batch_size, instances_per_epoch=max_data_points)(
                task.val_data, num_epochs=1, shuffle=False)
            n_val_batches = math.ceil(max_data_points / batch_size)
            all_val_metrics["%s_loss" % task.name] = 0.0

            for batch in val_generator:
                batch_num += 1
                out = self._forward(batch, task=task, for_training=False)
                loss = out["loss"]
                all_val_metrics["%s_loss" % task.name] += loss.data.cpu().numpy()
                n_examples += out["n_exs"]

                # log
                if time.time() - task_info['last_log'] > self._log_interval:
                    task_metrics = task.get_metrics()
                    task_metrics["%s_loss" % task.name] = \
                        all_val_metrics["%s_loss" % task.name] / batch_num
                    description = self._description_from_metrics(task_metrics)
                    log.info("Batch %d/%d: %s", batch_num, n_val_batches, description)
                    task_info['last_log'] = time.time()
            assert batch_num == n_val_batches

            # Get task validation metrics and store in all_val_metrics
            task_metrics = task.get_metrics(reset=True)
            for name, value in task_metrics.items():
                all_val_metrics["%s_%s" % (task.name, name)] = value
            all_val_metrics["%s_loss" % task.name] /= batch_num  # n_val_batches
            if task.val_metric_decreases and len(tasks) > 1:
                all_val_metrics["micro_avg"] += (1 - all_val_metrics[task.val_metric] /
                                                 self._dec_val_scale) * n_examples
                all_val_metrics["macro_avg"] += (1 -
                                                 all_val_metrics[task.val_metric] /
                                                 self._dec_val_scale)
            else:
                # triggers for single-task cases and during MTL when task val metric increases
                all_val_metrics["micro_avg"] += all_val_metrics[task.val_metric] * n_examples
                all_val_metrics["macro_avg"] += all_val_metrics[task.val_metric]
            n_examples_overall += n_examples

            # Reset training progress
            task_info['n_batches_since_val'] = 0
            task_info['loss'] = 0

        all_val_metrics['micro_avg'] /= n_examples_overall
        all_val_metrics['macro_avg'] /= len(tasks)

        # Track per task patience
        should_save = periodic_save  # whether to save this epoch or not.
        # Currently we save every validation in the main training runs.
        new_best_macro = False  # whether this epoch is a new best

        for task in tasks + ['micro', 'macro']:
            if task in ['micro', 'macro']:
                metric = "%s_avg" % task
                metric_decreases = tasks[0].val_metric_decreases if len(tasks) == 1 else False
                task_name = task
            else:
                metric = task.val_metric
                metric_decreases = task.val_metric_decreases
                task_name = task.name
            if metric_infos[metric]['stopped']:
                continue
            this_epoch_metric = all_val_metrics[metric]
            metric_history = metric_infos[metric]['hist']
            metric_history.append(this_epoch_metric)
            is_best_so_far, out_of_patience = \
                self._check_history(metric_history, this_epoch_metric, metric_decreases)
            if is_best_so_far:
                log.info("Best model found for %s.", task_name)
                metric_infos[metric]['best'] = (epoch, all_val_metrics)
                should_save = True
                if task_name == 'macro':
                    new_best_macro = True
            if out_of_patience:
                if periodic_save:
                    should_save = True
                metric_infos[metric]['stopped'] = True
                log.info("Out of patience. Stopped tracking %s", task_name)

            # Get scheduler, using global scheduler if exists and task is macro
            # micro has no scheduler updates
            if task_name not in ['micro', 'macro'] and g_scheduler is None:
                scheduler = task_infos[task_name]['scheduler']
            elif g_scheduler is not None and task_name == 'macro':
                scheduler = g_scheduler
            else:
                scheduler = None
            if scheduler is not None and isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                log.info("Advancing scheduler.")
                scheduler.step(this_epoch_metric, epoch)
                log.info("\tBest %s: %.3f", metric, scheduler.lr_scheduler.best)
                log.info("\t# bad epochs: %d", scheduler.lr_scheduler.num_bad_epochs)

        return all_val_metrics, should_save, new_best_macro

    def _get_lr(self):
        ''' Get learning rate from the optimizer we're using '''
        if self._g_optimizer is not None:
            lrs = {'global_lr': self._g_optimizer.param_groups[0]['lr']}
        else:
            lrs = {}
            for task, task_info in self._task_infos.items():
                lrs["%s_lr" % task] = task_info['optimizer'].param_groups[0]['lr']
        return lrs

    def _check_stop(self, epoch, stop_metric, tasks):
        ''' Check to see if should stop '''
        task_infos, metric_infos = self._task_infos, self._metric_infos
        g_optimizer = self._g_optimizer

        if self._max_epochs_per_task > 0:
            for task in tasks:
                task_info = task_infos[task.name]
                n_epochs_trained = task_info['total_batches_trained'] / task_info['n_tr_batches']
                if n_epochs_trained >= self._max_epochs_per_task:
                    log.info("Maximum batches trained on %s.", task.name)
                    task_info['stopped'] = True

        if g_optimizer is None:
            stop_tr = True
            for task in tasks:
                task_info = task_infos[task.name]
                if task_info['optimizer'].param_groups[0]['lr'] < self._min_lr:
                    log.info("Minimum lr hit on %s.", task.name)
                    task_info['stopped'] = True
                stop_tr = stop_tr and task_info['stopped']
        else:
            stop_tr = False
            if g_optimizer.param_groups[0]['lr'] < self._min_lr:
                log.info("Minimum lr hit.")
                stop_tr = True
            if min([info["stopped"] for info in task_infos.values()]):
                log.info("Maximum batches trained on all tasks.")
                stop_tr = True

        stop_val = metric_infos[stop_metric]['stopped']

        should_stop = False
        if stop_tr:
            should_stop = True
            log.info("All tasks hit minimum lr. Stopping training.")
        if stop_val:
            should_stop = True
            log.info("All metrics ran out of patience. Stopping training.")
        if epoch >= self._max_vals:
            log.info("Maximum number of validations hit. Stopping training.")
            should_stop = True

        return should_stop

    def _forward(self, batch, for_training, task=None):
        ''' At one point this does something, now it doesn't really do anything '''
        tensor_batch = move_to_device(batch, self._cuda_device)
        model_out = self._model.forward(task, tensor_batch)
        return model_out

    def _description_from_metrics(self, metrics):
        # pylint: disable=no-self-use
        ''' format some metrics as a string '''
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _unmark_previous_best(self, phase, epoch):
        marked_best = glob.glob(
            os.path.join(self._serialization_dir, "*_state_{}_epoch_*.best_macro.th".format(phase)))
        for file in marked_best:
            # Skip the just-written checkpoint.
            if "_{}.".format(epoch) not in file:
                os.rename(file, re.sub('%s$' % ".best_macro.th", ".th", file))

    def _delete_old_checkpoints(self, phase, epoch):
        candidates = glob.glob(
            os.path.join(self._serialization_dir, "*_state_{}_epoch_*.th".format(phase)))
        for file in candidates:
            # Skip the best, because we'll need it.
            # Skip the just-written checkpoint.
            if ".best_macro" not in file and "_{}.".format(epoch) not in file:
                os.remove(file)

    def _save_checkpoint(self, training_state, phase="main", new_best_macro=False, keep_all=False):
        """
        Parameters
        ----------
        training_state: An object containing trainer state (step number, etc.), to be saved.
        phase: Usually 'main' or 'eval'.
        new_best_macro: If true, the saved checkpoint will be marked with .best_macro, and
            potentially used later when switching from main to eval training.
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        epoch = training_state["epoch"]
        if phase == "eval":
            model_path = os.path.join(
                self._serialization_dir,
                "model_state_eval_best.th")
        else:
            if new_best_macro:
                best_str = ".best_macro"
            else:
                best_str = ""

            model_path = os.path.join(
                self._serialization_dir,
                "model_state_{}_epoch_{}{}.th".format(
                    phase, epoch, best_str))

        model_state = self._model.state_dict()

        # Skip non-trainable params, like the main ELMo params.
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                del model_state[name]
        torch.save(model_state, model_path)

        if phase != "eval":
            torch.save(
                training_state,
                os.path.join(
                    self._serialization_dir,
                    "training_state_{}_epoch_{}{}.th".format(
                        phase, epoch, best_str)))

            task_states = {}
            for task_name, task_info in self._task_infos.items():
                task_states[task_name] = {}
                task_states[task_name]['total_batches_trained'] = task_info['total_batches_trained']
                task_states[task_name]['stopped'] = task_info['stopped']
                if self._g_optimizer is None:
                    task_states[task_name]['optimizer'] = task_info['optimizer'].state_dict()
                    sched_params = {}
                    task_states[task_name]['scheduler'] = sched_params
            task_states['global'] = {}
            task_states['global']['optimizer'] = self._g_optimizer.state_dict() if \
                self._g_optimizer is not None else None
            if self._g_scheduler is not None:
                sched_params = {}
                task_states['global']['scheduler'] = sched_params
            else:
                task_states['global']['scheduler'] = None
            torch.save(task_states, os.path.join(self._serialization_dir,
                                                 "task_state_{}_epoch_{}{}.th".format(
                                                     phase, epoch, best_str)))

            metric_states = {}
            for metric_name, metric_info in self._metric_infos.items():
                metric_states[metric_name] = {}
                metric_states[metric_name]['hist'] = metric_info['hist']
                metric_states[metric_name]['stopped'] = metric_info['stopped']
                metric_states[metric_name]['best'] = metric_info['best']
            torch.save(
                metric_states,
                os.path.join(
                    self._serialization_dir,
                    "metric_state_{}_epoch_{}{}.th".format(
                        phase, epoch, best_str)))
        log.info("Saved files to %s", self._serialization_dir)

        if phase != "eval" and new_best_macro:
            self._unmark_previous_best(phase, epoch)

        if not self._keep_all_checkpoints:
            self._delete_old_checkpoints(phase, epoch)

    def _find_last_checkpoint_suffix(self, search_phases_in_priority_order=['main']):
        """
        Search for checkpoints to load, looking only for `main` training checkpoints.

        TODO: This is probably hairier than it needs to be. If you're good at string handling...
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        for current_search_phase in search_phases_in_priority_order:
            max_epoch = 0
            to_return = None
            candidate_files = glob.glob(
                os.path.join(
                    self._serialization_dir,
                    "model_state_{}_*".format(current_search_phase)))
            for x in candidate_files:
                epoch = int(x.split("model_state_{}_epoch_".format(
                    current_search_phase))[-1].split(".")[0])
                if epoch >= max_epoch:
                    max_epoch = epoch
                    to_return = x
            return to_return.split("model_state_")[-1]

    def _restore_checkpoint(self, search_phases_in_priority_order=['main']):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch
            The epoch at which to resume training.
        """

        suffix_to_load = self._find_last_checkpoint_suffix(
            search_phases_in_priority_order=search_phases_in_priority_order)
        assert suffix_to_load, "No checkpoint found."
        log.info("Found checkpoint {}. Loading.".format(suffix_to_load))

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_{}".format(suffix_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_{}".format(suffix_to_load))
        task_state_path = os.path.join(self._serialization_dir,
                                       "task_state_{}".format(suffix_to_load))
        metric_state_path = os.path.join(self._serialization_dir,
                                         "metric_state_{}".format(suffix_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))

        for name, param in self._model.named_parameters():
            if param.requires_grad and name not in model_state:
                log.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                log.error("Parameter missing from checkpoint: " + name)
                log.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self._model.load_state_dict(model_state, strict=False)

        task_states = torch.load(task_state_path)
        for task_name, task_state in task_states.items():
            if task_name == 'global':
                continue
            self._task_infos[task_name]['total_batches_trained'] = task_state['total_batches_trained']
            if 'optimizer' in task_state:
                self._task_infos[task_name]['optimizer'].load_state_dict(task_state['optimizer'])
                for param, val in task_state['scheduler'].items():
                    setattr(self._task_infos[task_name]['scheduler'], param, val)
            self._task_infos[task_name]['stopped'] = task_state['stopped']
            generator = self._task_infos[task_name]['tr_generator']
            for _ in itertools.islice(generator, task_state['total_batches_trained'] %
                                      self._task_infos[task_name]['n_tr_batches']):
                pass
        if task_states['global']['optimizer'] is not None:
            self._g_optimizer.load_state_dict(task_states['global']['optimizer'])
        if task_states['global']['scheduler'] is not None:
            for param, val in task_states['global']['scheduler'].items():
                setattr(self._g_scheduler, param, val)

        metric_states = torch.load(metric_state_path)
        for metric_name, metric_state in metric_states.items():
            self._metric_infos[metric_name]['hist'] = metric_state['hist']
            self._metric_infos[metric_name]['stopped'] = metric_state['stopped']
            self._metric_infos[metric_name]['best'] = metric_state['best']

        training_state = torch.load(training_state_path)
        return training_state["pass"], training_state["should_stop"]

    def _metrics_to_tensorboard_tr(self, epoch, train_metrics, task_name):
        """
        Sends all of the train metrics to tensorboard
        """
        metric_names = train_metrics.keys()

        for name in metric_names:
            train_metric = train_metrics.get(name)
            name = task_name + '/' + task_name + '_' + name
            self._TB_train_log.add_scalar(name, train_metric, epoch)

    def _metrics_to_tensorboard_val(self, epoch, val_metrics):
        """
        Sends all of the val metrics to tensorboard
        """
        metric_names = val_metrics.keys()

        for name in metric_names:
            val_metric = val_metrics.get(name)
            name = name.split('_')[0] + '/' + name
            self._TB_validation_log.add_scalar(name, val_metric, epoch)

    @classmethod
    def from_params(cls, model, serialization_dir, params):
        ''' Generate trainer from parameters.  '''

        patience = params.pop("patience", 2)
        val_interval = params.pop("val_interval", 100)
        max_vals = params.pop("max_vals", 50)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        keep_all_checkpoints = params.pop("keep_all_checkpoints", False)
        val_data_limit = params.pop("val_data_limit", 5000)
        max_epochs_per_task = params.pop("max_epochs_per_task", -1)
        dec_val_scale = params.pop("dec_val_scale", 100)
        training_data_fraction = params.pop("training_data_fraction", 1.0)

        params.assert_empty(cls.__name__)
        return SamplingMultiTaskTrainer(model, patience=patience,
                                        val_interval=val_interval, max_vals=max_vals,
                                        serialization_dir=serialization_dir,
                                        cuda_device=cuda_device, grad_norm=grad_norm,
                                        grad_clipping=grad_clipping, lr_decay=lr_decay,
                                        min_lr=min_lr,
                                        keep_all_checkpoints=keep_all_checkpoints,
                                        val_data_limit=val_data_limit,
                                        max_epochs_per_task=max_epochs_per_task,
                                        dec_val_scale=dec_val_scale,
                                        training_data_fraction=training_data_fraction)
