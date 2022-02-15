import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os
import benchmark_submission_formatter
from distutils.dir_util import copy_tree
import shutil
import json
from make_plots import create

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# Dict to huggingface name of the model or local directory containing model (sloberta)
pretrained_models = {
    # "sloberta": "./models/pretrained/sloberta",
    "crosloengual": "EMBEDDIA/crosloengual-bert",
    # "multilingual": "bert-base-multilingual-cased",
    # "roberta": "roberta-base",
    # "xlm-roberta-base": 'xlm-roberta-base'
}


pretrained = pretrained_models["crosloengual"]  # model to transformers model
output_name = list(pretrained_models.keys())[list(pretrained_models.values()).index(pretrained)]  # name of the model
name = "slobench"  # name of the directory containing datasets and config files

tasks = ["wsc"]  # list of tasks - can also be ["boolq", "cb", "copa", "multirc", "rte", "wsc"] for multitask

# name of output directory
if len(tasks) == 1:
    task_name = tasks[0]
else:
    task_name = "multitask_"
    for i in tasks:
        task_name = f"{task_name}{i}_"

# Here is where we set batch size and number of epochs and declare whether we want to use GPU or not. In the phases
# list we declare which phase should the tokenizer tokenize.
train_batch_size = 1
eval_batch_size = 1
epochs = 10
num_gpus = 1
phases = ["train", "val", "test", "val_test"]

# Here we set if we want to plot a graph based on validation results and loss and number of steps after which we
# want to check. If graph per epoch is True then this script will calculate the number of steps in one epoch and
# make validations on every epoch
eval_every_epoch = True
graph_steps = 1
graph_per_epoch = True
save_every_epoch = True  # set true if you want to save model after epoch eval
epochs_to_save = []  # if we do eval on every epoch, we can after which epoch we want to save the model during training
# if we want to save after all epoch set list to [i+1 for i in range(epochs)]. Note that one save equals to 0.5GB

# Some tokenization parameters
warmup_steps_proportion = 0.1
max_seq_length = 256
smart_truncate = True
do_iter = True

# Runscript parameters.
learning_rate = 1e-5
optimizer_type = "adam"
adam_epsilon = 1e-8
max_grad_norm = 1.0
eval_every_steps = 0
no_improvements_for_n_evals = 0
eval_subset_num = None
model_load_mode = "from_transformers"  # If we wish to load saved model from jiant we have to set model_load_mode to "partial"

do_train = True
do_val = True
validate_test = True
force_overwrite = True
write_test_preds = True
write_val_preds = True
write_val_test_preds = True
do_save = False
do_save_best = True
do_save_last = False
load_best_model = True

# This will export the hugging face model to directory "./models/name_of_model"
export_model.export_model(
    hf_pretrained_model_name_or_path=pretrained,
    output_base_path=f"./models/{output_name}",
)

# Tokenize and cache each task
for task_name in tasks:
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"./tasks/configs/{name}/{task_name}_config.json",
        hf_pretrained_model_name_or_path=pretrained,
        output_dir=f"./cache/{task_name}",
        phases=phases,
        do_iter=do_iter,
        smart_truncate=smart_truncate,
        max_seq_length=max_seq_length,
    ))

# Make configuration
jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"./tasks/configs/{name}",
    task_cache_base_path="./cache",
    train_task_name_list=tasks,
    val_task_name_list=tasks,
    test_task_name_list=tasks,
    val_test_task_name_list=tasks,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    epochs=epochs,
    num_gpus=num_gpus,
    eval_subset_num=eval_subset_num,
    warmup_steps_proportion=warmup_steps_proportion,
).create_config()

# Make directories and place configuration .json file
os.makedirs("./run_configs/", exist_ok=True)
os.makedirs(f"./runs/{output_name}/{name}", exist_ok=True)
py_io.write_json(jiant_run_config, "./run_configs/jiant_run_config.json")
display.show_json(jiant_run_config)

# If we want to make evaluation after each epoch we overwrite graph_steps to be equal as the number of steps in one epoch
if graph_per_epoch:
    with open("./run_configs/jiant_run_config.json", "r") as json_file:
        json_f = json.load(json_file)
        max_steps = json_f["global_train_config"]["max_steps"]
    graph_steps = max_steps // epochs

with open("./run_configs/jiant_run_config.json", "r") as json_file:
    json_f = json.load(json_file)
    max_steps = json_f["global_train_config"]["max_steps"]
epoch_steps = max_steps // epochs

# Run configuration
run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path="./run_configs/jiant_run_config.json",
    output_dir=f"./runs/{output_name}/{name}",
    model_load_mode=model_load_mode,
    hf_pretrained_model_name_or_path=pretrained,
    #  model_path="./runs/sloberta/human_translation/best_model.p", -> example if we want to load trained model from jiant
    model_path=f"./models/{output_name}/model/model.p",
    model_config_path=f"./models/{output_name}/model/config.json",
    learning_rate=learning_rate,
    eval_every_steps=eval_every_steps,
    do_train=do_train,
    do_val=do_val,
    do_val_test=validate_test,
    force_overwrite=force_overwrite,
    write_test_preds=write_test_preds,
    write_val_preds=write_val_preds,
    write_val_test_preds=write_val_test_preds,
    do_save_best=do_save_best,
    do_save_last=do_save_last,
    do_save=do_save,
    no_cuda=True if not num_gpus else False,
    no_improvements_for_n_evals=no_improvements_for_n_evals,
    adam_epsilon=adam_epsilon,
    max_grad_norm=max_grad_norm,
    optimizer_type=optimizer_type,
    load_best_model=load_best_model,
    graph_steps=graph_steps,
    graph_per_epoch=graph_per_epoch,
    epoch_steps=epoch_steps,
    epochs_to_save=epochs_to_save,
    save_every_epoch=save_every_epoch
)

main_runscript.run_loop(run_args)

if do_train and eval_every_epoch:
    create(tasks=tasks, path_to_look="./runs", num_epochs=epochs, model_name=output_name, epoch_length=epoch_steps)


# if we want to write predictions to file on test dataset
if "test" in phases and write_test_preds:
    benchmark_submission_formatter.results(
        benchmark="SUPERGLUE",
        input_base_path=f"./runs/{output_name}/{name}",
        output_path=f"./runs/{output_name}/{name}",
        task_names=tasks,
        preds="test_preds.p",
        regime="test",
    )

# if we want to write predictions to file on val_test dataset
if "val_test" in phases and write_val_preds:
    benchmark_submission_formatter.results(
        benchmark="SUPERGLUE",
        input_base_path=f"./runs/{output_name}/{name}",
        output_path=f"./runs/{output_name}/{name}",
        task_names=tasks,
        preds="val_test_preds.p",
        regime="val_test",
    )

# if we want to write predictions to file on val dataset
if "val" in phases and do_val:
    benchmark_submission_formatter.results(
        benchmark="SUPERGLUE",
        input_base_path=f"./runs/{output_name}/{name}",
        output_path=f"./runs/{output_name}/{name}",
        task_names=tasks,
        preds="val_preds.p",
        regime="val",
    )

# script makes output in ./run directory and if we run multiple times it will overwrite. You can copy to a new directory
# and copy the output to new directory if do_backup is set to True

do_backup = True

if do_backup:
    bak_folder = f"./trained_models/{output_name}__{name}__{task_name}__epochs_{epochs}__train_batch_{train_batch_size}__eval_batch_{eval_batch_size}__num_eval_steps_{eval_every_steps}"

    if os.path.isdir(bak_folder):
        shutil.rmtree(bak_folder)

    os.makedirs(bak_folder)
    os.makedirs(f"{bak_folder}/run_configs")

    copy_tree("./runs", bak_folder)
    copy_tree("./run_configs", f"{bak_folder}/run_configs")

    shutil.rmtree("./runs")
    shutil.rmtree("./run_configs")
    shutil.rmtree(f"./models/{output_name}")
    shutil.rmtree("./cache")





