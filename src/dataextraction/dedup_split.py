#!/usr/bin/env python
"""
Remove near duplicates from data and perform train/test/validation/holdout splits.

Usage:
    dedup_split.py [options] INPUT_FILENAME OUTPUT_FOLDER

Arguments:
    INPUT_FOLDER               directory w/ compressed jsonl files that have a .jsonl.gz a file extension
    OUTPUT_FOLDER              directory where you want to save data to.

Options:
    -h --help                    Show this screen.
    --azure-info=<path>          Azure authentication information file (JSON).
    --train-ratio FLOAT          Ratio of files for training set. [default: 0.6]
    --valid-ratio FLOAT          Ratio of files for validation set. [default: 0.15]
    --test-ratio FLOAT           Ratio of files for test set. [default: 0.15]
    --holdout-ratio FLOAT        Ratio of files for test set. [default: 0.1]
    --debug                      Enable debug routines. [default: False]

Example:

    python dedup_split.py \
    --azure-info /ds/hamel/azure_auth.json \
    azure://semanticcodesearch/pythondata/raw_v2  \
    azure://semanticcodesearch/pythondata/Processed_Data_v2

"""

from docopt import docopt
import hashlib
import pandas as pd
from utils.pkldf2jsonl import chunked_save_df_to_jsonl
from dpu_utils.utils import RichPath, run_and_debug
from dpu_utils.codeutils.deduplication import DuplicateDetector
import os
from tqdm import tqdm


def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)


def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    "Resolve near duplicates based upon code_tokens field in data."
    assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
    assert 'language' in df.columns.values, 'Data must contain field language'
    df.reset_index(inplace=True, drop=True)
    df['doc_id'] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=10)
    filter_mask = df.apply(lambda x: dd.add_file(id=x.doc_id,
                                                 tokens=x.code_tokens,
                                                 language=x.language),
                           axis=1)
    # compute fuzzy duplicates
    exclusion_set = dd.compute_ids_to_exclude()
    # compute pandas.series of type boolean which flags whether or not code should be discarded
    # in order to resolve duplicates (discards all but one in each set of duplicate functions)
    exclusion_mask = df['doc_id'].apply(lambda x: x not in exclusion_set)

    # filter the data
    print(f'Removed {sum(~(filter_mask & exclusion_mask)):,} fuzzy duplicates out of {df.shape[0]:,} rows.')
    return df[filter_mask & exclusion_mask]


def label_folds(df: pd.DataFrame, train_ratio: float, valid_ratio: float, test_ratio: float, holdout_ratio: float) -> pd.DataFrame:
    "Adds a partition column to DataFrame with values: {train, valid, test, holdout}."
    assert abs(train_ratio + valid_ratio + test_ratio + holdout_ratio - 1) < 1e-5,  'Ratios must sum up to 1.'
    # code in the same file will always go to the same split
    df['hash_key'] = df.apply(lambda x: f'{x.repo}:{x.path}', axis=1)
    df['hash_val'] = df['hash_key'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (2**16))

    train_bound = int(2**16 * train_ratio)
    valid_bound = train_bound + int(2**16 * valid_ratio)
    test_bound = valid_bound + int(2**16 * test_ratio)

    def label_splits(hash_val: int) -> str:
        if hash_val <= train_bound:
            return "train"
        elif hash_val <= valid_bound:
            return "valid"
        elif hash_val <= test_bound:
            return "test"
        else:
            return "holdout"

    # apply partition logic
    df['partition'] = df['hash_val'].apply(lambda x: label_splits(x))
    # display summary statistics
    counts = df.groupby('partition')['repo'].count().rename('count')
    summary_df = pd.concat([counts, (counts / counts.sum()).rename('pct')], axis=1)
    print(summary_df)

    return df


def run(args):

    azure_info_path = args.get('--azure-info', None)
    input_path = RichPath.create(args['INPUT_FILENAME'], azure_info_path)
    output_folder = RichPath.create(args['OUTPUT_FOLDER'], azure_info_path)
    train = float(args['--train-ratio'])
    valid = float(args['--valid-ratio'])
    test = float(args['--test-ratio'])
    holdout = float(args['--holdout-ratio'])

    # get data and process it
    df = jsonl_to_df(input_path)
    print('Removing fuzzy duplicates ... this may take some time.')
    df = remove_duplicate_code_df(df)
    df = df.sample(frac=1, random_state=20181026)  # shuffle order of files
    df = label_folds(df, train_ratio=train, valid_ratio=valid, test_ratio=test, holdout_ratio=holdout)
    splits = ['train', 'valid', 'test', 'holdout']

    for split in splits:
        split_df = df[df.partition == split]

        # save dataframes as chunked jsonl files
        jsonl_save_folder = output_folder.join(f'jsonl/{split}')
        print(f'Uploading data to {str(jsonl_save_folder)}')
        chunked_save_df_to_jsonl(split_df, jsonl_save_folder)

        # Upload dataframes to Azure
        filename = f'/tmp/{split}_df.pkl'
        df_save_path = output_folder.join(f'DataFrame/{split}_df.pkl')
        split_df.to_pickle(filename)
        print(f'Uploading data to {str(df_save_path)}')
        df_save_path.copy_from(RichPath.create(filename))
        os.unlink(filename)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug'))
