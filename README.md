This repository is a fork of the jiantv2 repository from https://github.com/nyu-mll/jiant/ with some modifications for evaluations
on Slovene version of SuperGLUE.

### Prerequisite:
 - Linux based OS
 - Python3.x

### Installation:

We recommend creation of a new virtual environment and install the dependencies in that environment. Alternatively you can
also use conda.

##### Steps:

- git clone "this_repo"
- cd jiant_slovene 
-  mkdir venv
- python3 -m venv ./venv
- source ./venv/bin/activate
- pip install -r requirements.txt
- pip install -e .

Once the environment is set you should create a new directory named "tasks" in the same directory as this project.
This directory will contain your datasets. See tasks_directory_structure.txt as an example on how to set the "tasks" directory for machine translated version
of SuperGlue. {task_name}_config.json must contain absolute paths to the directories that contain dataset files
(train.jsonl, val.jsonl, test.jsonl, test_with_answers.jsonl). For example boolq_config.json for machine translated SuperGLUE:
```
{"task": "boolq", "paths": {"train": "/home/matic/Desktop/graphs_jiant/tasks/data/machine_translation/boolq/train.jsonl", "val": "/home/matic/Desktop/graphs_jiant/tasks/data/machine_translation/boolq/val.jsonl", "test": "/home/matic/Desktop/graphs_jiant/tasks/data/machine_translation/boolq/test.jsonl", "test_with_answers": "/home/matic/Desktop/graphs_jiant/tasks/data/machine_translation/boolq/test_with_answers.jsonl"}, "name": "boolq"}
```
All directories and tasks must be in lower case. For easier creation of config files see script "config_file.py".


#### Running main.py
Once the directory is set you can change the parameters (for example batch size, epochs etc.) in main.py and run it.

The script first loads all the necessary libraries and then reads the dataset (caches it and saves). Then it creates a
configuration and starts training.

Variable "pretrained" should contain a string with the path to transformers model for example "EMBEDDIA/crosloengual-bert".
There is a bug in this version of transformers that this repo uses and one has to manually add Sloberta model to directory
if needed for evaluations. Sloberta can be downloaded from https://www.clarin.si/repository/xmlui/handle/11356/1397 and extracted
to "./models/pretrained/sloberta" directory. Variable "output_name" (name of the output model) is taken from the dictionary containing paths to models - if someone
wants to evaluate another model it should be added to dictionary or manually assign "output_name".
List "tasks" is a list of strings and contains the tasks we want to evaluate (for single task add only one string for example "cb",
for multitask add multiple strings like "cb", "boolq" etc.)

We have added two features (still needs to be tested) to evaluate after each epoch on development dataset (val.jsonl) and
create a plot visualizing accuracy per epoch and loss per epoch. To do that set parameter "eval_every_epoch=True"
and the number of steps in epoch "graph_steps={some number}" to desired number. If you do not know the number of steps in one epoch
set variable "graph_per_epoch=True" and the script will calculate how many steps are there in one epoch. We also added the
option to save the model every each epoch if we are also validating after each epoch. To save after epoch set variable "save_every_epoch=True"
and insert numbers of after which epoch you wish to save the model (for example "epochs_to_save=[1, 4, 10]" will save the model
after first, fourth and tenth epoch). The default output of each run is in "./runs" directory and the models are overwritten
if we execute the script again. At the end of main.py you can see a variable "do_backup=True" which will copy
the ./runs" directory to:
```
"f"./trained_models/{output_name}__{name}__{task_name}__epochs_{epochs}__train_batch_{train_batch_size}__eval_batch_{eval_batch_size}__num_eval_steps_{eval_every_steps}"
```
This can be disabled or modified according to the needs of anyone.

We have also added the "test_with_answers" phase, which can be run at the end of training. This "test_with_answers.jsonl" contains the solved "test.jsonl"
examples of a task and it calculates the metrics on those examples.

The saved models can be used again to train or simply just to do evaluation. If you wish to use a saved model
you have to set variable " model_load_mode='partial' " instead of 'from_transformers' and "model_path" in "main_runscript.RunConfiguration" set to the path of the saved model.
For clarification see "main_eval_saved_model.py". If you do not wish to train the model again set variable "do_train=False".

After each run there are some file that are created (in "./runs" or in the backup directory) that contain information about the run
like val_metrics.json and test_with_answers_metrics.json contain the metrics of evaluations on "val" and "test_with_answers" phase, loss_train.zlog contains
information about the loss after each step etc.

We recommend a quick run of the main.py (for example epoch=0.1) and see for yourself all the output files jiant creates.

You can also read the original README.md below for extra clarification or see the example notebooks in ./examples.

_________________________________________________________________________________________
# Original README.md


<div align="center">

# `jiant` is an NLP toolkit
**The multitask and transfer learning toolkit for natural language processing research**

[![Generic badge](https://img.shields.io/github/v/release/nyu-mll/jiant)](https://shields.io/)
[![codecov](https://codecov.io/gh/nyu-mll/jiant/branch/master/graph/badge.svg)](https://codecov.io/gh/nyu-mll/jiant)
[![CircleCI](https://circleci.com/gh/nyu-mll/jiant/tree/master.svg?style=shield)](https://circleci.com/gh/nyu-mll/jiant/tree/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

**Why should I use `jiant`?**
- `jiant` supports [multitask learning](https://colab.research.google.com/github/nyu-mll/jiant/blob/master/examples/notebooks/jiant_Multi_Task_Example.ipynb)
- `jiant` supports [transfer learning](https://colab.research.google.com/github/nyu-mll/jiant/blob/master/examples/notebooks/jiant_STILTs_Example.ipynb)
- `jiant` supports [50+ natural language understanding tasks](./guides/tasks/supported_tasks.md)
- `jiant` supports the following benchmarks:
    - [GLUE](./guides/benchmarks/glue.md)
    - [SuperGLUE](./guides/benchmarks/superglue.md)
    - [XTREME](./guides/benchmarks/xtreme.md)
- `jiant` is a research library and users are encouraged to extend, change, and contribute to match their needs!

**A few additional things you might want to know about `jiant`:**
- `jiant` is configuration file driven
- `jiant` is built with [PyTorch](https://pytorch.org)
- `jiant` integrates with [`datasets`](https://github.com/huggingface/datasets) to manage task data
- `jiant` integrates with [`transformers`](https://github.com/huggingface/transformers) to manage models and tokenizers.

## Getting Started

* Get started with some simple [Examples](./examples)
* Learn more about `jiant` by reading our [Guides](./guides)
* See our [list of supported tasks](./guides/tasks/supported_tasks.md)

## Installation

To import `jiant` from source (recommended for researchers):
```bash
git clone https://github.com/nyu-mll/jiant.git
cd jiant
pip install -r requirements.txt

# Add the following to your .bash_rc or .bash_profile 
export PYTHONPATH=/path/to/jiant:$PYTHONPATH
```
If you plan to contribute to jiant, install additional dependencies with `pip install -r requirements-dev.txt`.

To install `jiant` from source (alternative for researchers):
```
git clone https://github.com/nyu-mll/jiant.git
cd jiant
pip install . -e
```

To install `jiant` from pip (recommended if you just want to train/use a model):
```
pip install jiant
```

We recommended that you install `jiant` in a virtual environment or a conda environment.

To check `jiant` was correctly installed, run a [simple example](./examples/notebooks/simple_api_fine_tuning.ipynb).


## Quick Introduction
The following example fine-tunes a RoBERTa model on the MRPC dataset.

Python version:
```python
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

# Download the Data
downloader.download_data(["mrpc"], "/content/data")

# Set up the arguments for the Simple API
args = run.RunConfiguration(
   run_name="simple",
   exp_dir="/path/to/exp",
   data_dir="/path/to/exp/tasks",
   model_type="roberta-base",
   tasks="mrpc",
   train_batch_size=16,
   num_train_epochs=3
)

# Run!
run.run_simple(args)
```

Bash version:
```bash
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks mrpc \
    --output_path /path/to/exp/tasks
python jiant/proj/simple/runscript.py \
    run \
    --run_name simple \
    --exp_dir /path/to/exp \
    --data_dir /path/to/exp/tasks \
    --model_type roberta-base \
    --tasks mrpc \
    --train_batch_size 16 \
    --num_train_epochs 3
```

Examples of more complex training workflows are found [here](./examples/).


## Contributing
The `jiant` project's contributing guidelines can be found [here](CONTRIBUTING.md).

## Looking for `jiant v1.3.2`?
`jiant v1.3.2` has been moved to [jiant-v1-legacy](https://github.com/nyu-mll/jiant-v1-legacy) to support ongoing research with the library. `jiant v2.x.x` is more modular and scalable than `jiant v1.3.2` and has been designed to reflect the needs of the current NLP research community. We strongly recommended any new projects use `jiant v2.x.x`.

`jiant 1.x` has been used in in several papers. For instructions on how to reproduce papers by `jiant` authors that refer readers to this site for documentation (including Tenney et al., Wang et al., Bowman et al., Kim et al., Warstadt et al.), refer to the [jiant-v1-legacy](https://github.com/nyu-mll/jiant-v1-legacy) README.

## Citation

If you use `jiant ≥ v2.0.0` in academic work, please cite it directly:

```
@misc{phang2020jiant,
    author = {Jason Phang and Phil Yeres and Jesse Swanson and Haokun Liu and Ian F. Tenney and Phu Mon Htut and Clara Vania and Alex Wang and Samuel R. Bowman},
    title = {\texttt{jiant} 2.0: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2020}
}
```

If you use `jiant ≤ v1.3.2` in academic work, please use the citation found [here](https://github.com/nyu-mll/jiant-v1-legacy).

## Acknowledgments

- This work was made possible in part by a donation to NYU from Eric and Wendy Schmidt made
by recommendation of the Schmidt Futures program, and by support from Intuit Inc.
- We gratefully acknowledge the support of NVIDIA Corporation with the donation of a Titan V GPU used at NYU in this work.
- Developer Jesse Swanson is supported by the Moore-Sloan Data Science Environment as part of the NYU Data Science Services initiative.

## License
`jiant` is released under the [MIT License](https://github.com/nyu-mll/jiant/blob/master/LICENSE).
