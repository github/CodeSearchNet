from collections import defaultdict
from itertools import chain
from typing import Optional, List, Dict, Any, NamedTuple, Iterable, Tuple
import logging
import random

from dpu_utils.mlutils import Vocabulary
from dpu_utils.utils import RichPath
import numpy as np
from more_itertools import chunked, flatten
from scipy.spatial.distance import cdist
import wandb

import model_restore_helper
from models.model import get_data_files_from_directory, Model
from dataextraction.python.parse_python_data import tokenize_python_from_string
from dataextraction.utils import tokenize_docstring_from_string
from dpu_utils.codeutils import split_identifier_into_parts


def compute_ranks(src_representations: np.ndarray,
                  tgt_representations: np.ndarray,
                  distance_metric: str) -> Tuple[np.array, np.array]:
    distances = cdist(src_representations, tgt_representations,
                      metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    return np.sum(distances <= correct_elements, axis=-1), distances


class MrrSearchTester:
    def __init__(self, model_path: RichPath, test_batch_size: int=1000, distance_metric: str='cosine',
                 quiet: bool=False, hypers_override: Optional[Dict[str, Any]]=None) -> None:
        self.__model = model_restore_helper.restore(path=model_path,
                                                    is_train=False,
                                                    hyper_overrides=hypers_override)
        self.__quiet = quiet
        self.__test_batch_size = test_batch_size
        self.__distance_metric = distance_metric

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def test_batch_size(self)-> int:
        return self.__test_batch_size

    def update_test_batch_size(self, test_batch_size: int)-> None:
        self.__test_batch_size = test_batch_size

    QueryResult = NamedTuple('QueryResult', [
        ('target_idx', int),
        ('target_rank', int),
        ('top_ranked_idxs', List[int])
    ])

    def evaluate(self, data: List[Dict[str, Any]], data_label_name: str,
                 error_log: Optional[List['MrrSearchTester.QueryResult']]=None,
                 error_log_rank_threshold: int=10,
                 filter_language: Optional[str]=None)-> float:
        """
        Evaluate the MRR on the given dataset.

        :param data: the data to test on.
        :param data_label_name: A label used when printing the result output.
        :param error_log: If not null, store in the log, results where the target rank is above the threshold.
        :param error_log_rank_threshold: The threshold for logging into error_log (used only if error_log is not None)
        :return: the mean reciprocal rank (MRR) score
        """
        assert len(data) > 0, 'data must have more than 0 rows.'
        np.random.seed(0)  # set random seed so that random things are reproducible

        if filter_language is None:
            idxs = np.arange(len(data))
        else:
            idxs = np.array([i for i in range(len(data)) if data[i]['language'] == filter_language])
        if len(idxs) == 0:
            print('Warning: Trying to test on empty dataset. Skipping.')
            return float('nan')
        data = np.array(data, dtype=np.object)
        np.random.shuffle(idxs)
        data = data[idxs]

        if len(data) < self.__test_batch_size:
            logging.warning(f'the size of the total data {len(data):,} is less than the batch_size: {self.__test_batch_size:,} adjusting batch size to equal data size.')
            self.update_test_batch_size(len(data))

        def self_or_random_representation(representation: Optional[np.ndarray]) -> np.ndarray:
            if representation is not None:
                return representation
            else:
                return np.random.randn(self.__model.representation_size)

        # Determine random sample of examples before chunking into batches.
        # sample only from full batches
        max_samples = 50
        full_batch_len = len(data) // self.__test_batch_size * self.__test_batch_size
        examples_sample = np.zeros(len(data), dtype=bool)
        examples_sample[np.random.choice(np.arange(full_batch_len), replace=False, size=min(full_batch_len, max_samples))] = True
        examples_table = []

        sum_mrr = 0.0
        num_batches = 0
        batched_data = chunked(data, self.__test_batch_size)
        batched_sample = chunked(examples_sample, self.__test_batch_size)
        for batch_idx, (batch_data, batch_sample) in enumerate(zip(batched_data, batched_sample)):
            if len(batch_data) < self.__test_batch_size:
                break  # the last batch is smaller than the others, exclude.
            num_batches += 1

            code_representations = self.__model.get_code_representations(batch_data)
            query_representations = self.__model.get_query_representations(batch_data)
            assert len(code_representations) == len(query_representations) == self.__test_batch_size

            # Construct numpy batch
            num_uncomputed_representations = sum(1 for i in range(self.__test_batch_size)
                                                 if code_representations[i] is None or query_representations[i] is None)
            if num_uncomputed_representations > 0:
                print(f'Ignoring {num_uncomputed_representations} samples whose representation could not be computed')

            # Design decision: If a representation cannot be computed assign a random representation. This keeps
            # the batch size identical across all models.
            batch_code_representations = np.array(
                [self_or_random_representation(code_representations[i]) for i in range(self.__test_batch_size)],
                dtype=np.float32)
            batch_query_representations = np.array(
                [self_or_random_representation(query_representations[i]) for i in range(self.__test_batch_size)],
                dtype=np.float32)

            ranks, distances = compute_ranks(batch_code_representations,
                                             batch_query_representations,
                                             self.__distance_metric)

            # Log example tables for a sample of rankings of queries for each dataset
            if wandb.run:
                examples_table_name = data_label_name.rstrip("-All")
                examples_table_columns = ["Rank", "Language", "Query", "Code"]
                for example, sample, rank in zip(batch_data, batch_sample, ranks):
                    if not sample:
                        continue
                    language = example['language']
                    markdown_code = "```%s\n" % language + example['code'].strip("\n") + "\n```"
                    examples_table.append([rank, language, example['func_name'], markdown_code])

            sum_mrr += np.mean(1.0 / ranks)

            if error_log is not None:
                batch_sample_idxs = idxs[batch_idx*self.__test_batch_size:(batch_idx+1)*self.__test_batch_size]
                for i in range(len(ranks)):
                    if ranks[i] >= error_log_rank_threshold:
                        result = MrrSearchTester.QueryResult(
                            target_idx=batch_sample_idxs[i],
                            target_rank=ranks[i],
                            top_ranked_idxs=batch_sample_idxs[np.argsort(distances[i])[:3]]
                        )
                        error_log.append(result)

            if self.__quiet and batch_idx % 100 == 99:
                print(f'Tested on {batch_idx + 1} batches so far.')

        if wandb.run and examples_table:
            wandb.log({"Examples-%s" % examples_table_name: wandb.Table(columns=examples_table_columns, rows=examples_table)})

        eval_mrr = sum_mrr / num_batches
        log_label = f'{data_label_name} MRR (bs={self.__test_batch_size:,})'
        print(f'{log_label}: {eval_mrr: .3f}')
        if wandb.run:
            wandb.run.summary[f'{log_label}'] = eval_mrr
        return eval_mrr


def expand_data_path(data_path: str, azure_info_path: Optional[str]) -> List[RichPath]:
    """
    Args:
        data_path: A path to either a file or a directory. If it's a file, we interpret it as a list of
            data directories.

    Returns:
        List of data directories (potentially just data_path)
    """
    data_rpath = RichPath.create(data_path, azure_info_path)

    if data_rpath.is_dir():
        return [data_rpath]

    return [RichPath.create(data_dir, azure_info_path)
            for data_dir in data_rpath.read_as_text().splitlines()]


def filter_untokenizable_code(data: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out data where field code_tokens is empty."""
    return [d for d in data if d['code_tokens']]

def log_row_count_diff(original_data: Iterable[Any], filtered_data:Iterable[Any], label: str) -> None:
    """Compute the difference between row counts and log appropriately."""
    original_row_count = len(list(original_data))
    filtered_row_count = len(list(filtered_data))

    assert original_row_count > 0, 'original_data does not contain any rows.'
    assert filtered_row_count <= original_row_count, f'filtered_data {filtered_row_count:,} has a larger row count than original_data {original_row_count:,}.'

    pcnt_parsed = filtered_row_count / original_row_count
    print(f'{label}: parsed {filtered_row_count:,} out of {original_row_count:,} rows. ({pcnt_parsed*100:.1f}%)')
    if wandb.run:
        wandb.run.summary[f'{label} Parsed Pct'] = pcnt_parsed


def get_dataset_from(data_dirs: List[RichPath], 
                     use_func_names: bool=False, 
                     max_files_per_dir: Optional[int] = None) -> List[Dict[str, Any]]:
    data_files = sorted(get_data_files_from_directory(data_dirs, max_files_per_dir))
    data = list(chain(*chain(list(f.read_by_file_suffix()) for f in data_files)))

    if use_func_names:
        # This task tries to match the function name to the code, by setting the function name as the query
        for sample in data:
            # Replace the query tokens with the function name, broken up into its sub-tokens:
            sample['docstring_tokens'] = split_identifier_into_parts(sample['func_name'])

            # In the code, replace the function name with the out-of-vocab token everywhere it appears:
            sample['code_tokens'] = [Vocabulary.get_unk() if token == sample['func_name'] else token
                                     for token in sample['code_tokens']]
    return data


def compute_evaluation_metrics(model_path: RichPath, arguments, 
                               azure_info_path: str,
                               valid_data_dirs: List[RichPath], 
                               test_data_dirs: List[RichPath],
                               max_files_per_dir: Optional[int] = None):

    tester = MrrSearchTester(model_path, test_batch_size=int(arguments['--test-batch-size']),
                                  distance_metric=arguments['--distance-metric'])
    test_data = get_dataset_from(test_data_dirs, max_files_per_dir=max_files_per_dir)
    # Get all languages in test_data
    dataset_languages = set(d['language'] for d in test_data)
    evaluation_sets = list((l, True) for l in dataset_languages)  # type: List[Tuple[str, bool]]
    if set(tester.model.per_code_language_metadata.keys()) == dataset_languages:
        evaluation_sets = [('All', False)] + evaluation_sets
    final_eval = {}  # type: Dict[str, float]
    for language_name, filter_language in evaluation_sets:
        if filter_language and language_name not in tester.model.per_code_language_metadata:
            continue
        mrr = tester.evaluate(test_data, f'Test-{language_name}', filter_language=language_name if filter_language else None)
        if language_name == "All":
            final_eval['Primary MRR'] = mrr

        # run test using the function name as the query
        mrr = tester.evaluate(get_dataset_from(test_data_dirs, use_func_names=True, max_files_per_dir=max_files_per_dir), f'FuncNameTest-{language_name}',
                              filter_language=language_name if filter_language else None)
        if language_name == "All":
            final_eval['FuncName MRR'] = mrr

        # run the test procedure on the validation set (with same batch size as test, so that MRR is comparable)
        tester.evaluate(get_dataset_from(valid_data_dirs, max_files_per_dir=max_files_per_dir), f'Validation-{language_name}',
                        filter_language=language_name if filter_language else None)

    if wandb.run and final_eval:
        wandb.run.summary['Eval'] = final_eval
