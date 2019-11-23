from collections import defaultdict
from typing import *

from dpu_utils.utils import RichPath

from encoders import Encoder
from utils.general_utils import get_data_files_from_directory
from utils.py_utils import run_jobs_in_parallel

__all__ = ("Metadata", "MetadataLoader")

Metadata = Dict[str, Any]


class MetadataLoader:
    def __init__(
            self,
            query_encoder_type: Type[Encoder],
            code_encoder_types: Callable[[str], Type[Encoder]],
            hyperparameters: Dict[str, Any],
    ):
        self.query_encoder_type = query_encoder_type
        self.code_encoder_types = code_encoder_types
        self.raw_query_metadata_list = []
        self.raw_code_language_metadata_lists: DefaultDict[str, list] = defaultdict(list)
        self.hyperparameters = hyperparameters

    def init_code_metadata(self, language: str):
        return self.code_encoder_types(language).init_metadata()

    def metadata_parser_fn(self, _, file_path: RichPath) -> Iterable[Tuple[Metadata, Dict[str, Metadata]]]:
        raw_query_metadata: Metadata = self.query_encoder_type.init_metadata()
        per_code_language_metadata: Dict[str, Metadata] = {}

        for raw_sample in file_path.read_by_file_suffix():
            sample_language = raw_sample['language']
            self.code_encoder_types(sample_language).load_metadata_from_sample(
                raw_sample['code_tokens'],
                per_code_language_metadata.setdefault(sample_language, self.init_code_metadata(sample_language)),
                self.hyperparameters['code_use_subtokens'],
                self.hyperparameters['code_mark_subtoken_end'],
            )
            self.query_encoder_type.load_metadata_from_sample(
                [d.lower() for d in raw_sample['docstring_tokens']],
                raw_query_metadata,
            )
        yield (raw_query_metadata, per_code_language_metadata)

    def received_result_callback(self, metadata_parser_result: Tuple[Metadata, Dict[str, Metadata]]):
        (raw_query_metadata, per_code_language_metadata) = metadata_parser_result
        self.raw_query_metadata_list.append(raw_query_metadata)
        for metadata_language, raw_code_language_metadata in per_code_language_metadata.items():
            self.raw_code_language_metadata_lists[metadata_language].append(raw_code_language_metadata)

    def load_parallel(self, data_dirs: List[RichPath], max_files_per_dir: Optional[int] = None):
        run_jobs_in_parallel(
            get_data_files_from_directory(data_dirs, max_files_per_dir),
            self.metadata_parser_fn,
            self.received_result_callback,
            noop,
        )

    def load_sequential(self, data_dirs: List[RichPath], max_files_per_dir: Optional[int] = None):
        for (idx, file) in enumerate(get_data_files_from_directory(data_dirs, max_files_per_dir)):
            for res in self.metadata_parser_fn(idx, file):
                self.received_result_callback(res)

    def finalize_metadata(self) -> Tuple[Metadata, Dict[str, Metadata]]:
        query_metadata = self.query_encoder_type.finalise_metadata("query", self.hyperparameters, self.raw_query_metadata_list)
        code_metadata = {}
        for language, raw_per_language_metadata in self.raw_code_language_metadata_lists.items():
            code_metadata[language] = self.code_encoder_types(language).finalise_metadata(
                "code", self.hyperparameters, raw_per_language_metadata
            )
        return query_metadata, code_metadata


def noop():
    pass
