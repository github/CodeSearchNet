#!/usr/bin/env python
"""
Usage:
    nearestneighbor.py [options] (--code | --query | --both) MODEL_PATH DATA_PATH

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --distance-metric METRIC   The distance metric to use [default: cosine]
    -h --help                  Show this screen.
    --hypers-override HYPERS   JSON dictionary overriding hyperparameter values.
    --debug                    Enable debug routines. [default: False]
    --num-nns NUM              The number of NNs to visualize [default: 2]
    --distance-threshold TH    The distance threshold above which to ignore [default: 0.2]
    --max-num-items LIMIT      The maximum number of items to use. Use zero for all. [default: 0]
"""
import json
from itertools import chain
from typing import Any, Dict, List

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
import numpy as np
from more_itertools import take
from scipy.spatial.distance import pdist

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

import model_restore_helper

# Condensed to square from
# http://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
from utils.visutils import square_to_condensed


def to_string(code: str, language: str) -> str:
    lexer = get_lexer_by_name(language, stripall=True)
    formatter = TerminalFormatter(linenos=True)
    return highlight(code, lexer, formatter)

def run(arguments) -> None:
    azure_info_path = arguments.get('--azure-info', None)
    data_path = RichPath.create(arguments['DATA_PATH'], azure_info_path)
    assert data_path.is_dir(), "%s is not a folder" % (data_path,)

    hypers_override = arguments.get('--hypers-override')
    if hypers_override is not None:
        hypers_override = json.loads(hypers_override)
    else:
        hypers_override = {}

    model_path = RichPath.create(arguments['MODEL_PATH'], azure_info_path=azure_info_path)

    model = model_restore_helper.restore(
        path=model_path,
        is_train=False,
        hyper_overrides=hypers_override)

    num_elements_to_take = int(arguments['--max-num-items'])
    data = chain(*chain(list(f.read_by_file_suffix()) for f in data_path.iterate_filtered_files_in_dir('*.jsonl.gz')))
    if num_elements_to_take == 0:  # Take all
        data = list(data)
    else:
        assert num_elements_to_take > 0
        data = take(num_elements_to_take, data)


    num_nns = int(arguments['--num-nns'])

    if arguments['--code']:
        representations = model.get_code_representations(data)
    elif arguments['--query']:
        representations = model.get_query_representations(data)
    else:
        code_representations = model.get_code_representations(data)
        query_representations = model.get_query_representations(data)
        representations = np.concatenate([code_representations, query_representations], axis=-1)

    filtered_representations = []
    filtered_data = []  # type: List[Dict[str, Any]]
    for i, representation in enumerate(representations):
        if representation is None:
            continue
        filtered_representations.append(representation)
        filtered_data.append(data[i])

    filtered_representations = np.stack(filtered_representations, axis=0)
    flat_distances = pdist(filtered_representations, arguments['--distance-metric'])

    for i, data in enumerate(filtered_data):
        distance_from_i = np.fromiter(
            (flat_distances[square_to_condensed(i, j, len(filtered_data))] if i != j else float('inf') for j in
             range(len(filtered_data))), dtype=np.float)

        nns = [int(k) for k in np.argsort(distance_from_i)[:num_nns]]  # The first two NNs

        if distance_from_i[nns[0]] > float(arguments['--distance-threshold']):
            continue

        print('===============================================================')
        print(f"{data['repo']}:{data['path']}:{data['lineno']}")
        print(to_string(data['original_string'], language=data['language']))

        for j in range(num_nns):
            print()
            print(f'Nearest Neighbour {j+1}: {filtered_data[nns[j]]["repo"]}:{filtered_data[nns[j]]["path"]}:{filtered_data[nns[j]]["lineno"]} (distance {distance_from_i[nns[j]]})')
            print(to_string(filtered_data[nns[j]]['original_string'], language=filtered_data[nns[j]]['language']))


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
