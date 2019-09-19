#!/usr/bin/env python
"""
Usage:
    error_analysis.py [options] MODEL_PATH (--standard-dataset | --method2code-dataset) DATA_PATH OUT_FILE

Options:
    -h --help                        Show this screen.
    --max-num-epochs EPOCHS          The maximum number of epochs to run [default: 300]
    --max-num-files INT              Number of files to load.
    --max-num-examples INT           Randomly sample examples from the dataset to display.
    --hypers-override HYPERS         JSON dictionary overriding hyperparameter values.
    --hypers-override-file FILE      JSON file overriding hyperparameter values.
    --test-batch-size SIZE           The size of the batches in which to compute MRR. [default: 1000]
    --distance-metric METRIC         The distance metric to use [default: cosine]
    --quiet                          Less output (not one per line per minibatch). [default: False]
    --azure-info PATH                Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                          Enable debug routines. [default: False]
    --standard-dataset               The DATA_PATH is to a standard dataset.
    --method2code-dataset            The DATA_PATH is to a standard dataset but will be used for method2code tasks.
    --language-to-analyze LANG       The language to analyze. Defaults to all.
"""
import io
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

import model_test
from model_test import expand_data_path, MrrSearchTester
from random import sample


## Default Bootstrap headers
HEADER=f"""
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

    <title>Error Analysis</title>
    
    <style>
        {HtmlFormatter().get_style_defs('.highlight')}
    </style>
  </head>
  <body>
"""
FOOTER="""
</body></html>
"""

def to_highlighted_html(code:str, language: str) -> str:
    lexer = get_lexer_by_name(language, stripall=True)
    formatter = HtmlFormatter(linenos=True)
    return highlight(code, lexer, formatter)

def generate_html_error_report(tester: MrrSearchTester,
                               data:  List[Dict[str, Any]],
                               max_num_examples: Optional[int],
                               outfile: str,
                               filter_language: Optional[str] = None) -> None:

    error_log = []  # type: List[MrrSearchTester.QueryResult]
    # Sample the data if requested
    data = sample_data(data=data,
                       max_num_examples=max_num_examples)

    # generate error logs
    tester.update_test_batch_size(max_num_examples)
    tester.evaluate(data, 'Error Analysis Run', error_log, filter_language=filter_language)

    "Generates HTML Report of Errors."
    print('Generating Report')
    with open(outfile, 'w') as f:
        f.write(HEADER)
        for query_result in tqdm(error_log, total=len(error_log)):
            with io.StringIO() as sb:
                target_code = data[query_result.target_idx]['code']
                target_query = data[query_result.target_idx]['docstring'].replace('\n', ' ')
                language = data[query_result.target_idx]['language']
                sb.write(f'<h2> Query: "{target_query}"</h2>\n\n')
                sb.write(f'<strong>Target Snippet</strong>\n{to_highlighted_html(target_code, language=language)}\n')
                sb.write(f'Target snippet was ranked at position <strong>{query_result.target_rank}</strong>.\n')

                sb.write('<div class="row">\n')
                for pos, sample_idx in enumerate(query_result.top_ranked_idxs):
                    sb.write('<div class="col-sm">\n')
                    sb.write(f'<strong>Result at {pos+1}</strong>\n')
                    sb.write(f'{data[sample_idx]["repo"]} {data[sample_idx]["path"]}:{data[sample_idx]["lineno"]}\n')
                    result_docstring = data[sample_idx]['docstring']
                    result_code = data[sample_idx]['code']
                    lang = data[sample_idx]['language']
                    sb.write(f'<blockquote><p>  Docstring: <em>{result_docstring}</em></blockquote>\n{to_highlighted_html(result_code, language=lang)}\n\n')
                    sb.write('</div>\n')
                sb.write('</div>\n<hr/>\n')
                f.write(sb.getvalue())
        f.write(FOOTER)


def sample_data(data: List[Dict[str, Any]],
                max_num_examples: Optional[int]) -> List[Dict[str, Any]]:
    """
    Sample max_num_examples from the data.

    Args:
        data: List[Dict[str, Any]]
        max_num_examples:  either an int or if a string will attempt conversion to an int.

    Returns:
        data: List[Dict[str, Any]]
    """
    if max_num_examples:
        num_elements = min(len(data), max_num_examples)
        print(f'Extracting {num_elements} random samples from dataset.')
        data = sample(data, num_elements)

    return data


def run(arguments):
    max_num_examples = int(arguments.get('--max-num-examples')) if arguments.get('--max-num-examples') else None
    azure_info_path = arguments.get('--azure-info', None)
    test_data_dirs = expand_data_path(arguments['DATA_PATH'], azure_info_path)

    if arguments['--hypers-override'] is not None:
        hypers_override = json.loads(arguments['--hypers-override'])
    elif arguments['--hypers-override-file'] is not None:
        with open(arguments['--hypers-override-file']) as f:
            hypers_override = json.load(f)
    else:
        hypers_override = {}

    model_path = RichPath.create(arguments['MODEL_PATH'], azure_info_path=azure_info_path)

    tester = MrrSearchTester(model_path, test_batch_size=int(arguments['--test-batch-size']),
                             distance_metric=arguments['--distance-metric'], hypers_override=hypers_override)

    # Load dataset
    if arguments['--standard-dataset'] or arguments['--method2code-dataset']:
        data = model_test.get_dataset_from(test_data_dirs, use_func_names=arguments['--method2code-dataset'])
    else:
        raise Exception(f'No dataset option seems to have been passed in.')

    generate_html_error_report(tester=tester,
                               data=data,
                               max_num_examples=max_num_examples,
                               outfile=arguments['OUT_FILE'],
                               filter_language=arguments.get('--language-to-analyze'))

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
