import multiprocessing
from collections import defaultdict
from typing import *

from dpu_utils.utils import RichPath

from encoders import Encoder
from utils.general_utils import get_data_files_from_directory

LoadedSamples = Dict[str, List[Dict[str, Any]]]


def parse_data_file(
        hyperparameters: Dict[str, Any],
        code_encoder_types: Callable[[str], Type[Encoder]],
        per_code_language_metadata: Dict[str, Dict[str, Any]],
        query_encoder_type: Type[Encoder],
        query_metadata: Dict[str, Any],
        is_test: bool,
        data_file: RichPath,
) -> Dict[str, List[Tuple[bool, Dict[str, Any]]]]:
    results: DefaultDict[str, List] = defaultdict(list)
    for raw_sample in data_file.read_by_file_suffix():
        sample: Dict = {}
        language = raw_sample['language']
        if language.startswith('python'):  # In some datasets, we use 'python-2.7' and 'python-3'
            language = 'python'

        # the load_data_from_sample method call places processed data into sample, and
        # returns a boolean flag indicating if sample should be used
        function_name = raw_sample.get('func_name')
        use_code_flag = code_encoder_types(language).load_data_from_sample(
            "code",
            hyperparameters,
            per_code_language_metadata[language],
            raw_sample['code_tokens'],
            function_name,
            sample,
            is_test,
        )
        use_query_flag = query_encoder_type.load_data_from_sample(
            "query",
            hyperparameters,
            query_metadata,
            [d.lower() for d in raw_sample['docstring_tokens']],
            function_name,
            sample,
            is_test,
        )
        use_example = use_code_flag and use_query_flag
        results[language].append((use_example, sample))
    return results


class DataLoader:
    def __init__(
            self,
            query_encoder_type: Type[Encoder],
            code_encoder_types: Callable[[str], Type[Encoder]],
            hyperparameters: Dict[str, Any],
            query_metadata: Dict[str, Any],
            code_metadata: Dict[str, Dict[str, Any]],
    ):
        self.query_encoder_type = query_encoder_type
        self.code_encoder_types = code_encoder_types
        self.hyperparameters = hyperparameters
        self.query_metadata = query_metadata
        self.code_metadata = code_metadata

    def parse_data_file(self, code_metadata, query_metadata, is_test, data_file):
        return parse_data_file(
            self.hyperparameters,
            self.code_encoder_types,
            code_metadata,
            self.query_encoder_type,
            query_metadata,
            is_test,
            data_file,
        )

    def load_data_from_files(
            self,
            data_files: Iterable[RichPath], *,
            is_test: bool,
            parallelize: bool = True,
    ) -> Tuple[LoadedSamples, int]:
        tasks_as_args = [
            (self.code_metadata, self.query_metadata, is_test, data_file)
            for data_file in data_files
        ]

        if parallelize:
            with multiprocessing.Pool() as pool:
                per_file_results = pool.starmap(self.parse_data_file, tasks_as_args)
        else:
            per_file_results = [self.parse_data_file(*task_args) for task_args in tasks_as_args]

        samples: DefaultDict[str, List] = defaultdict(list)
        num_all_samples = 0
        for per_file_result in per_file_results:
            for (language, parsed_samples) in per_file_result.items():
                for (use_example, parsed_sample) in parsed_samples:
                    num_all_samples += 1
                    if use_example:
                        samples[language].append(parsed_sample)
        return samples, num_all_samples

    def load_data_from_dirs(
            self,
            data_dirs: List[RichPath], *,
            is_test: bool,
            max_files_per_dir: Optional[int] = None,
            parallelize: bool = True,
    ) -> Tuple[LoadedSamples, int]:
        return self.load_data_from_files(
            list(get_data_files_from_directory(data_dirs, max_files_per_dir)),
            is_test=is_test,
            parallelize=parallelize,
        )
