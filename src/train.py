#!/usr/bin/env python
"""
Usage:
    train.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH
    train.py [options] [SAVE_FOLDER]

*_DATA_PATH arguments may either accept (1) directory filled with .jsonl.gz files that we use as data,
or a (2) plain text file containing a list of such directories (used for multi-language training).

In the case that you supply a (2) plain text file, all directory names must be separated by a newline.
For example, if you want to read from multiple directories you might have a plain text file called
data_dirs_train.txt with the below contents:

> cat ~/src/data_dirs_train.txt
azure://semanticcodesearch/pythondata/Processed_Data/jsonl/train
azure://semanticcodesearch/csharpdata/split/csharpCrawl-train

Options:
    -h --help                        Show this screen.
    --max-num-epochs EPOCHS          The maximum number of epochs to run [default: 300]
    --max-files-per-dir INT          Maximum number of files per directory to load for training data.
    --hypers-override HYPERS         JSON dictionary overriding hyperparameter values.
    --hypers-override-file FILE      JSON file overriding hyperparameter values.
    --model MODELNAME                Choose model type. [default: neuralbowmodel]
    --test-batch-size SIZE           The size of the batches in which to compute MRR. [default: 1000]
    --distance-metric METRIC         The distance metric to use [default: cosine]
    --run-name NAME                  Picks a name for the trained model.
    --quiet                          Less output (not one per line per minibatch). [default: False]
    --dryrun                         Do not log run into logging database. [default: False]
    --testrun                        Do a model run on a small dataset for testing purposes. [default: False]
    --azure-info PATH                Azure authentication information file (JSON). Used to load data from Azure storage.
    --evaluate-model PATH            Run evaluation on previously trained model.
    --sequential                     Do not parallelise data-loading. Simplifies debugging. [default: False]
    --debug                          Enable debug routines. [default: False]
"""
import json
import os
import sys
import time
from typing import Type, Dict, Any, Optional, List
from pathlib import Path

from docopt import docopt
from dpu_utils.utils import RichPath, git_tag_run, run_and_debug
import wandb

import model_restore_helper
from model_test import compute_evaluation_metrics
from models.model import Model
import model_test as test

def run_train(model_class: Type[Model],
              train_data_dirs: List[RichPath],
              valid_data_dirs: List[RichPath],
              save_folder: str,
              hyperparameters: Dict[str, Any],
              azure_info_path: Optional[str],
              run_name: str,
              quiet: bool = False,
              max_files_per_dir: Optional[int] = None,
              parallelize: bool = True) -> RichPath:
    model = model_class(hyperparameters, run_name=run_name, model_save_dir=save_folder, log_save_dir=save_folder)
    if os.path.exists(model.model_save_path):
        model = model_restore_helper.restore(RichPath.create(model.model_save_path), is_train=True)
        model.train_log("Resuming training run %s of model %s with following hypers:\n%s" % (run_name,
                                                                                             model.__class__.__name__,
                                                                                             str(hyperparameters)))
        resume = True
    else:
        model.train_log("Tokenizing and building vocabulary for code snippets and queries.  This step may take several hours.")
        model.load_metadata(train_data_dirs, max_files_per_dir=max_files_per_dir, parallelize=parallelize)
        model.make_model(is_train=True)
        model.train_log("Starting training run %s of model %s with following hypers:\n%s" % (run_name,
                                                                                             model.__class__.__name__,
                                                                                             str(hyperparameters)))
        resume = False

    philly_job_id = os.environ.get('PHILLY_JOB_ID')
    if philly_job_id is not None:
        # We are in Philly write out the model name in an auxiliary file
        with open(os.path.join(save_folder, philly_job_id+'.job'), 'w') as f:
            f.write(os.path.basename(model.model_save_path))
    
    wandb.config.update(model.hyperparameters)
    model.train_log("Loading training and validation data.")
    train_data = model.load_data_from_dirs(train_data_dirs, is_test=False, max_files_per_dir=max_files_per_dir, parallelize=parallelize)
    valid_data = model.load_data_from_dirs(valid_data_dirs, is_test=False, max_files_per_dir=max_files_per_dir, parallelize=parallelize)
    model.train_log("Begin Training.")
    model_path = model.train(train_data, valid_data, azure_info_path, quiet=quiet, resume=resume)
    return model_path


def make_run_id(arguments: Dict[str, Any]) -> str:
    """Choose a run ID, based on the --save-name parameter, the PHILLY_JOB_ID and the current time."""
    philly_id = os.environ.get('PHILLY_JOB_ID')
    if philly_id is not None:
        return philly_id
    user_save_name = arguments.get('--run-name')
    if user_save_name is not None:
        user_save_name = user_save_name[:-len('.pkl')] if user_save_name.endswith('.pkl') else user_save_name
    else:
        user_save_name = arguments['--model']
    return "%s-%s" % (user_save_name, time.strftime("%Y-%m-%d-%H-%M-%S"))


def run(arguments, tag_in_vcs=False) -> None:
    azure_info_path = arguments.get('--azure-info', None)
    testrun = arguments.get('--testrun')
    max_files_per_dir=arguments.get('--max-files-per-dir')

    dir_path = Path(__file__).parent.absolute()

    # if you do not pass arguments for train/valid/test data default to files checked into repo.
    if not arguments['TRAIN_DATA_PATH']:
        arguments['TRAIN_DATA_PATH'] = str(dir_path/'data_dirs_train.txt')
        arguments['VALID_DATA_PATH'] = str(dir_path/'data_dirs_valid.txt')
        arguments['TEST_DATA_PATH'] = str(dir_path/'data_dirs_test.txt')

    train_data_dirs = test.expand_data_path(arguments['TRAIN_DATA_PATH'], azure_info_path)
    valid_data_dirs = test.expand_data_path(arguments['VALID_DATA_PATH'], azure_info_path)
    test_data_dirs = test.expand_data_path(arguments['TEST_DATA_PATH'], azure_info_path)
    
    # default model save location
    if not arguments['SAVE_FOLDER']:
        arguments['SAVE_FOLDER'] =  str(dir_path.parent/'resources/saved_models/')

    save_folder = arguments['SAVE_FOLDER']

    model_class = model_restore_helper.get_model_class_from_name(arguments['--model'])

    hyperparameters = model_class.get_default_hyperparameters()
    run_name = make_run_id(arguments)

    # make name of wandb run = run_id (Doesn't populate yet)
    hyperparameters['max_epochs'] = int(arguments.get('--max-num-epochs'))

    if testrun:
        hyperparameters['max_epochs'] = 2
        if not max_files_per_dir:
            max_files_per_dir = 1

    # override hyperparams if flag is passed
    hypers_override = arguments.get('--hypers-override')
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))
    elif arguments.get('--hypers-override-file') is not None:
        with open(arguments.get('--hypers-override-file')) as f:
            hyperparameters.update(json.load(f))

    os.makedirs(save_folder, exist_ok=True)

    if tag_in_vcs:
        hyperparameters['git_commit'] = git_tag_run(run_name)

    # turns off wandb if you don't want to log anything
    if arguments.get('--dryrun'):
        os.environ["WANDB_MODE"] = 'dryrun'
    # save hyperparams to logging
    # must filter out type=set from logging when as that is not json serializable
    wandb.init(name=run_name, config={k: v for k, v in hyperparameters.items() if not isinstance(v, set)})
    wandb.config.update({'model-class': arguments['--model'],
                         'train_folder': str(train_data_dirs),
                         'valid_folder': str(valid_data_dirs),
                         'save_folder': str(save_folder),
                         'test_folder': str(test_data_dirs),
                         'CUDA_VISIBLE_DEVICES': os.environ.get("CUDA_VISIBLE_DEVICES", 'Not Set'),
                         'run-name': arguments.get('--run-name'),
                         'CLI-command': ' '.join(sys.argv)})


    if arguments.get('--evaluate-model'):
        model_path = RichPath.create(arguments['--evaluate-model'])
    else:
        model_path = run_train(model_class, train_data_dirs, valid_data_dirs, save_folder, hyperparameters,
                               azure_info_path, run_name, arguments['--quiet'],
                               max_files_per_dir=max_files_per_dir,
                               parallelize=not(arguments['--sequential']))

    wandb.config['best_model_path'] = str(model_path)
    wandb.save(str(model_path.to_local_path()))

    # only limit files in test run if `--testrun` flag is passed by user.
    if testrun:
        compute_evaluation_metrics(model_path, arguments, azure_info_path, valid_data_dirs, test_data_dirs, max_files_per_dir)
    else:
        compute_evaluation_metrics(model_path, arguments, azure_info_path, valid_data_dirs, test_data_dirs)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
