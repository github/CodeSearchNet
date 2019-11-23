from typing import *
import pickle

import numpy as np
import pandas as pd
from dpu_utils.utils import RichPath


def save_file_pickle(fname: str, obj: Any) -> None:
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_file_pickle(fname: str) -> None:
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
        return obj


def chunkify(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    "turn pandas.dataframe into equal size n chunks."
    return [df[i::n] for i in range(n)]


def get_data_files_from_directory(
        data_dirs: List[RichPath],
        max_files_per_dir: Optional[int] = None,
) -> List[RichPath]:
    files = []  # type: List[RichPath]
    for data_dir in data_dirs:
        dir_files = data_dir.get_filtered_files_in_dir('*.jsonl.gz')
        if max_files_per_dir:
            dir_files = sorted(dir_files)[:int(max_files_per_dir)]
        files += dir_files

    np.random.shuffle(files)  # This avoids having large_file_0, large_file_1, ... subsequences
    return files
