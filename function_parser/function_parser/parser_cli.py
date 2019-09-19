"""
Usage:
    parser_cli.py [options] INPUT_FILEPATH

Options:
    -h --help
    --language LANGUAGE             Language
"""
import json

from docopt import docopt
from tree_sitter import Language

from language_data import LANGUAGE_METADATA
from process import DataProcessor

if __name__ == '__main__':
    args = docopt(__doc__)

    DataProcessor.PARSER.set_language(Language('/src/build/py-tree-sitter-languages.so', args['--language']))
    processor = DataProcessor(language=args['--language'],
                              language_parser=LANGUAGE_METADATA[args['--language']]['language_parser'])

    functions = processor.process_single_file(args['INPUT_FILEPATH'])
    print(json.dumps(functions, indent=2))
