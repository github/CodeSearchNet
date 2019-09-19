#!/usr/bin/env python
"""
Acquires python data from local disk or GCP and performs parsing, cleaning and tokenization steps
to form a parallel corpus of (code, docstring) pairs with additional metadata.  Processed data
is saved as multi-part jsonl files to the OUTPUT_PATH.

Usage:
    parse_python_data.py [options] OUTPUT_PATH

Options:
    -h --help                  Show this screen.
    --input-folder=<path>      Use the given input folder instead of downloading.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage
    --debug                    Enable debug routines. [default: False]

"""
import re
import os
from multiprocessing import Pool
from typing import List, NamedTuple

import pandas as pd
import parso
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from tqdm import tqdm

from dataextraction.utils import tokenize_docstring_from_string
from utils.pkldf2jsonl import chunked_save_df_to_jsonl


IS_WHITESPACE_REGEX = re.compile(r'\s+')


class ParsedCode(NamedTuple):
    code_tokens: List[str]
    comment_tokens: List[str]


def tokenize_python_from_string(code: str,
                                func_only: bool=True,
                                report_errors: bool=False,
                                only_ids: bool=False,
                                add_keywords: bool=True) -> ParsedCode:
    """
    Tokenize Python code given a string.

    Args:
        code: The input code
        func_only: if you want to only parse functions in code.
        report_errors: Flag that turns on verbose error reporting
        only_ids: Return only the identifiers within the code
        add_keywords: Return keywords (used only when only_ids=True)

    Returns:
        Pair of lists. First list is sequence of code tokens; second list is sequence of tokens in comments.
    """
    try:
        try:
            parsed_ast = parso.parse(code, error_recovery=False, version="2.7")
        except parso.parser.ParserSyntaxError:
            parsed_ast = parso.parse(code, error_recovery=False, version="3.7")
        code_tokens, comment_tokens = [], []

        func_nodes = list(parsed_ast.iter_funcdefs())

        # parse arbitrary snippets of code that are not functions if func_only = False
        if not func_only:
            func_nodes = [parsed_ast]
        
        for func_node in func_nodes:  # There should only be one, but we can process more...
            doc_node = func_node.get_doc_node()
            leaf_node = func_node.get_first_leaf()
            while True:
                # Skip over the docstring:
                if leaf_node is doc_node:
                    leaf_node = leaf_node.get_next_leaf()

                # First, retrieve comment tokens:
                for prefix in leaf_node._split_prefix():
                    if prefix.type == 'comment':
                        comment_text = prefix.value[1:]  # Split off the leading "#"
                        comment_tokens.extend(tokenize_docstring_from_string(comment_text))

                # Second, stop if we've reached the end:
                if leaf_node.type == 'endmarker':
                    break

                # Third, record code tokens:
                if not(IS_WHITESPACE_REGEX.match(leaf_node.value)):
                    if only_ids:
                        if leaf_node.type == 'name':
                            code_tokens.append(leaf_node.value)
                    else:
                        if leaf_node.type == 'keyword':
                            if add_keywords:
                                code_tokens.append(leaf_node.value)
                        else:
                            code_tokens.append(leaf_node.value)
                leaf_node = leaf_node.get_next_leaf()
        return ParsedCode(code_tokens=code_tokens, comment_tokens=comment_tokens)
    except Exception as e:
        if report_errors:
            print('Error tokenizing: %s' % (e,))
        return ParsedCode(code_tokens=[], comment_tokens=[])


def download_files_into_pandas(i: int=10) -> pd.DataFrame:
    """Get files from Google Cloud Platform, there are 10 files.

    Args:
        i : int between 1 and 10 that specifies how many of the 10 files you
            want to download.  You should only use this argument for testing.


    Files are obtained by this query: https://console.cloud.google.com/bigquery?sq=235037502967:58a5d62f75f34d22b0f70d38b9352a85
    """
    frames = []
    for i in tqdm(range(i), total=i):
        success = False
        while not success:
            try:
                frame = pd.read_csv(f'https://storage.googleapis.com/kubeflow-examples/code_search_new/python_raw_v2/00000000000{i}.csv', encoding='utf-8')
                frames.append(frame)
                success = True
            except Exception as e:
                print(f'Error downloading file {i}:\n {e}, retrying...')

    df = pd.concat(frames)

    df['repo'] = df['repo_path'].apply(lambda r: r.split()[0])
    df['path'] = df['repo_path'].apply(lambda r: r.split()[1])
    df.drop(columns=['repo_path'], inplace=True)
    df = df[['repo', 'path', 'content']]
    return df


def load_files_into_pandas(input_folder: str) -> pd.DataFrame:
    """Get files from a local directory.

    Args:
        input_folder: the folder containing the .csv files
    """
    frames = []
    for file in os.listdir(input_folder):
        if not file.endswith('.csv'):
            continue
        frame = pd.read_csv(os.path.join(input_folder, file), encoding='utf-8')
        frames.append(frame)

    df = pd.concat(frames)

    df['repo'] = df['repo_path'].apply(lambda r: r.split()[0])
    df['path'] = df['repo_path'].apply(lambda r: r.split()[1])
    df.drop(columns=['repo_path'], inplace=True)
    df = df[['repo', 'path', 'content']]
    return df


def parse_raw_data_into_function_list(blob, require_docstring: bool=True):
    """Extract per-function data from a given code blob.

    Filters out undesirable function types. Keep only the first line of the docstring, and remove all internal comments from
    the code.

    Args:
        blob: String containing some python code.

    Returns:
        List of functions represented by dictionaries containing the code, docstring and metadata.
    """
    parsed_data_list = []
    try:
        try:
            parsed_module = parso.parse(blob, error_recovery=False, version="2.7")
        except parso.parser.ParserSyntaxError:
            parsed_module = parso.parse(blob, error_recovery=False, version="3.7")

        function_defs = list(parsed_module.iter_funcdefs())
        for class_def in parsed_module.iter_classdefs():
            function_defs.extend(class_def.iter_funcdefs())

        for function_def in function_defs:
            function_name = function_def.name.value
            docstring_node = function_def.get_doc_node()
            if docstring_node is None:
                docstring = ''
            else:
                docstring = docstring_node.value
            first_docstring_line = docstring.split('\n\s*\n')[0]

            # We now need to un-indent the code which may have come from a class. For that, identify how far
            # we are indented, and try to to remove that from all lines:
            function_code = function_def.get_code()
            def_prefix = list(function_def.get_first_leaf()._split_prefix())[-1].value
            trimmed_lines = []
            for line in function_code.splitlines():
                if line.startswith(def_prefix):
                    trimmed_lines.append(line[len(def_prefix):])
            function_code = '\n'.join(trimmed_lines)

            should_use_function = not (re.search(r'(__.+__)|(.*test.*)|(.*Test.*)', function_name) or  # skip __*__ methods and test code
                                       re.search(r'NotImplementedException|@abstractmethod', function_code) or
                                       len(function_code.split('\n')) <= 2 or  # should have more than 1 line of code (the declaration is one line)
                                       (len(first_docstring_line.split()) <= 2) and require_docstring)  # docstring should have at least 3 words.

            if should_use_function:
                parsed_data_list.append({'code': function_code,
                                         'docstring': first_docstring_line,
                                         'language': 'python',
                                         'lineno': function_def.start_pos[0],
                                         'func_name': function_name,
                                         })

    except parso.parser.ParserSyntaxError:
        pass
    return parsed_data_list


def listlen(x):
    if not isinstance(x, list):
        return 0
    return len(x)


def run(args):
    azure_info_path = args.get('--azure-info')
    output_folder = RichPath.create(args['OUTPUT_PATH'], azure_info_path)

    # Download / read the data files:
    if args['--input-folder'] is None:
        print('Downloading data...')
        raw_code_data_df = download_files_into_pandas()
    else:
        print('Loading data...')
        raw_code_data_df = load_files_into_pandas(args['--input-folder'])
    print('Data loaded.')

    # Find all the functions and methods, filter out ones that don't meet requirements,
    # separate the code from the docstring and produce a list of functions that includes the code,
    # the first line of the docstring, and metadata of each:
    with Pool() as pool:
        function_data = pool.map(parse_raw_data_into_function_list, raw_code_data_df.content.tolist())
    assert len(function_data) == raw_code_data_df.shape[0], \
        f'Row count mismatch. `raw_code_data_df` has {raw_code_data_df.shape[0]} rows; `function_data` has {len(function_data)} rows.'
    raw_code_data_df['function_data'] = function_data
    print(f'Split {raw_code_data_df.shape[0]} blobs into {sum(len(fun_data) for fun_data in function_data)} documented individual functions.')

    # Flatten function data out:
    # TODO: We should also have access to the SHA of the objects here.
    raw_code_data_df = raw_code_data_df.set_index(['repo', 'path'])['function_data'].apply(pd.Series).stack()
    raw_code_data_df = raw_code_data_df.reset_index()
    raw_code_data_df.columns = ['repo', 'path', '_', 'function_data']

    # Extract meta-data and format dataframe.
    function_data_df = pd.DataFrame(raw_code_data_df.function_data.values.tolist())
    assert len(raw_code_data_df) == len(function_data_df), \
        f'Row count mismatch. `raw_code_data_df` has {len(raw_code_data_df)} rows; `function_data_df` has {len(function_data_df)} rows.'
    function_data_df = pd.concat([raw_code_data_df[['repo', 'path']], function_data_df], axis=1)

    # remove observations where the same code appears more than once
    num_before_dedup = len(function_data_df)
    function_data_df = function_data_df.drop_duplicates(['code'])
    num_after_dedup = len(function_data_df)

    print(f'Removed {num_before_dedup - num_after_dedup} exact duplicate rows.')

    print('Tokenizing code, comments and docstrings ...')
    with Pool() as pool:
        code_tokenization_results: List[ParsedCode] = pool.map(tokenize_python_from_string,
                                                               function_data_df['code'].tolist())

        code_tokens_list, comment_tokens_list = list(zip(*code_tokenization_results))
        function_data_df['code_tokens'] = code_tokens_list
        function_data_df['comment_tokens'] = comment_tokens_list
        function_data_df['docstring_tokens'] = pool.map(tokenize_docstring_from_string,
                                                        function_data_df['docstring'].tolist())
    function_data_df.dropna(subset=['code_tokens', 'comment_tokens', 'docstring_tokens'], inplace=True)
    function_data_df.reset_index(inplace=True, drop=True)

    cols_to_keep = ['repo', 'path', 'lineno', 'func_name', 'language',
                    'code', 'code_tokens', 'comment_tokens',
                    'docstring', 'docstring_tokens',
                    ]
    # write data to jsonl
    print(f'Count by language:\n{function_data_df.language.value_counts()}')
    chunked_save_df_to_jsonl(df=function_data_df[cols_to_keep],
                             output_folder=output_folder,
                             parallel=True)
    print(f'Wrote {function_data_df.shape[0]} rows to {str(output_folder)}.')


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug'))
