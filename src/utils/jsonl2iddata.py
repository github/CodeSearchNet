#!/usr/bin/env python
"""A utility tool for extracting the identifier tokens from existing .jsonl.gz data. Primarily used for exporting
data for MSR's tool for dataset deduplication at https://github.com/Microsoft/near-duplicate-code-detector.

Usage:
    jsonl2iddata.py [options] INPUT_PATH OUTPUT_PATH

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""
from docopt import docopt

from dpu_utils.utils import run_and_debug, RichPath, ChunkWriter


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    input_folder = RichPath.create(arguments['INPUT_PATH'], azure_info_path)
    output_folder = RichPath.create(arguments['OUTPUT_PATH'], azure_info_path)

    with ChunkWriter(output_folder, file_prefix='codedata', max_chunk_size=500, file_suffix='.jsonl.gz') as chunked_writer:
        for file in input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'):
            for line in file.read_by_file_suffix():
                tokens=line['code_tokens']
                chunked_writer.add(dict(filename='%s:%s:%s' % (line['repo'], line['path'], line['lineno']), tokens=tokens))


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
