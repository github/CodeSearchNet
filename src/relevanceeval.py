#!/usr/bin/env python
"""
Usage:
    relevanceeval.py [options] RELEVANCE_ANNOTATIONS_CSV_PATH MODEL_PREDICTIONS_CSV

Standalone relevance evaluation script that outputs evaluation statistics for a set of predictions of a given model.
The input formats of the files is described below.

The model predictions MODEL_PREDICTIONS_CSV file has the following format:
    A comma-separated file with (at least) the fields and headers "language", "query", "url". Each row represents
    a single result for a given query and a given programming language.

    * language: the programming language for the given query, e.g. "python"
    * query: the textual representation of the query, e.g. "int to string"
    * url: the unique GitHub URL to the returned results, e.g. "https://github.com/JamesClonk/vultr/blob/fed59ad207c9bda0a5dfe4d18de53ccbb3d80c91/cmd/commands.go#L12-L190"

     The order of the rows imply the ranking of the results in the search task. For example, if in row 5 there is
     an entry for the Python query "read properties file" and then in row 60 appears another result for the
     Python query "read properties file", then the URL in row 5 is considered to be ranked higher than the
     URL in row 60 for that query and language.

Options:
    --debug                          Run in debug mode, falling into pdb on exceptions.
    -h --help                        Show this screen.
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from docopt import docopt
from dpu_utils.utils import run_and_debug

def load_relevances(filepath: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    relevance_annotations = pd.read_csv(filepath)
    per_query_language = relevance_annotations.pivot_table(
        index=['Query', 'Language', 'GitHubUrl'], values='Relevance', aggfunc=np.mean)

    # Map language -> query -> url -> float
    relevances = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, float]]]
    for (query, language, url), relevance in per_query_language['Relevance'].items():
        relevances[language.lower()][query.lower()][url] = relevance
    return relevances

def load_predictions(filepath: str, max_urls_per_language: int=300) -> Dict[str, Dict[str, List[str]]]:
    prediction_data = pd.read_csv(filepath)

    # Map language -> query -> Ranked List of URL
    predictions = defaultdict(lambda: defaultdict(list))
    for _, row in prediction_data.iterrows():
        predictions[row['language'].lower()][row['query'].lower()].append(row['url'])
    for query_data in predictions.values():
        for query, ranked_urls in query_data.items():
            query_data[query] = ranked_urls[:max_urls_per_language]

    return predictions

def coverage_per_language(predictions: Dict[str, List[str]],
                          relevance_scores: Dict[str, Dict[str, float]], with_positive_relevance: bool=False) -> float:
    """
    Compute the % of annotated URLs that appear in the algorithm's predictions.
    """
    num_annotations = 0
    num_covered = 0
    for query, url_data in relevance_scores.items():
        urls_in_predictions = set(predictions[query])
        for url, relevance in url_data.items():
            if not with_positive_relevance or relevance > 0:
                num_annotations += 1
                if url in urls_in_predictions:
                    num_covered += 1

    return num_covered / num_annotations

def ndcg(predictions: Dict[str, List[str]], relevance_scores: Dict[str, Dict[str, float]],
         ignore_rank_of_non_annotated_urls: bool=True) -> float:
    num_results = 0
    ndcg_sum = 0

    for query, query_relevance_annotations in relevance_scores.items():
        current_rank = 1
        query_dcg = 0
        for url in predictions[query]:
            if url in query_relevance_annotations:
                query_dcg += (2**query_relevance_annotations[url] - 1) / np.log2(current_rank + 1)
                current_rank += 1
            elif not ignore_rank_of_non_annotated_urls:
                current_rank += 1

        query_idcg = 0
        for i, ideal_relevance in enumerate(sorted(query_relevance_annotations.values(), reverse=True), start=1):
            query_idcg += (2 ** ideal_relevance - 1) / np.log2(i + 1)
        if query_idcg == 0:
            # We have no positive annotations for the given query, so we should probably not penalize anyone about this.
            continue
        num_results += 1
        ndcg_sum += query_dcg / query_idcg
    return ndcg_sum / num_results



def run(arguments):
    relevance_scores = load_relevances(arguments['RELEVANCE_ANNOTATIONS_CSV_PATH'])
    predictions = load_predictions(arguments['MODEL_PREDICTIONS_CSV'])

    languages_predicted = sorted(set(predictions.keys()))

    # Now Compute the various evaluation results
    print('% of URLs in predictions that exist in the annotation dataset:')
    for language in languages_predicted:
        print(f'\t{language}: {coverage_per_language(predictions[language], relevance_scores[language])*100:.2f}%')

    print('% of URLs in predictions that exist in the annotation dataset (avg relevance > 0):')
    for language in languages_predicted:
        print(f'\t{language}: {coverage_per_language(predictions[language], relevance_scores[language], with_positive_relevance=True) * 100:.2f}%')

    print('NDCG:')
    for language in languages_predicted:
        print(f'\t{language}: {ndcg(predictions[language], relevance_scores[language]):.3f}')

    print('NDCG (full ranking):')
    for language in languages_predicted:
        print(f'\t{language}: {ndcg(predictions[language], relevance_scores[language], ignore_rank_of_non_annotated_urls=False):.3f}')



if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])