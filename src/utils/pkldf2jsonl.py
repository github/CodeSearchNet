import pandas as pd
from .general_utils import chunkify
from dpu_utils.utils import RichPath
from multiprocessing import Pool, cpu_count


def df_to_jsonl(df: pd.DataFrame, RichPath_obj: RichPath, i: int, basefilename='codedata') -> str:
    dest_filename = f'{basefilename}_{str(i).zfill(5)}.jsonl.gz'
    RichPath_obj.join(dest_filename).save_as_compressed_file(df.to_dict(orient='records'))
    return str(RichPath_obj.join(dest_filename))


def chunked_save_df_to_jsonl(df: pd.DataFrame,
                             output_folder: RichPath,
                             num_chunks: int=None,
                             parallel: bool=True) -> None:
    "Chunk DataFrame (n chunks = num cores) and save as jsonl files."

    df.reset_index(drop=True, inplace=True)
    # parallel saving to jsonl files on azure
    n = cpu_count() if num_chunks is None else num_chunks
    dfs = chunkify(df, n)
    args = zip(dfs, [output_folder]*len(dfs), range(len(dfs)))

    if not parallel:
        for arg in args:
            dest_filename = df_to_jsonl(*arg)
            print(f'Wrote chunk to {dest_filename}')
    else:
        with Pool(cpu_count()) as pool:
            pool.starmap(df_to_jsonl, args)

