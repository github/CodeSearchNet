#!/usr/bin/env python
"""
Usage:
    test.py [options] MODEL_PATH VALID_DATA_PATH TEST_DATA_PATH
    test.py [options] MODEL_PATH

Standalone testing script

Options:
    -h --help                        Show this screen.
    --test-batch-size SIZE           The size of the batches in which to compute MRR. [default: 1000]
    --distance-metric METRIC         The distance metric to use [default: cosine]
    --run-name NAME                  Picks a name for the trained model.
    --quiet                          Less output (not one per line per minibatch). [default: False]
    --dryrun                         Do not log run into logging database. [default: False]
    --azure-info PATH                Azure authentication information file (JSON). Used to load data from Azure storage.
    --sequential                     Do not parallelise data-loading. Simplifies debugging. [default: False]
    --debug                          Enable debug routines. [default: False]
"""
from pathlib import Path

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

import model_test as test


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)

    # if you do not pass arguments for train/valid/test data default to files checked into repo.
    if not arguments['VALID_DATA_PATH']:
        dir_path = Path(__file__).parent.absolute()
        print(dir_path)
        arguments['VALID_DATA_PATH'] = str(dir_path / 'data_dirs_valid.txt')
        arguments['TEST_DATA_PATH'] = str(dir_path / 'data_dirs_test.txt')

    valid_data_dirs = test.expand_data_path(arguments['VALID_DATA_PATH'], azure_info_path)
    test_data_dirs = test.expand_data_path(arguments['TEST_DATA_PATH'], azure_info_path)
    test.compute_evaluation_metrics(RichPath.create(arguments['MODEL_PATH'], azure_info_path=azure_info_path),
                                    arguments, azure_info_path, valid_data_dirs, test_data_dirs)

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])