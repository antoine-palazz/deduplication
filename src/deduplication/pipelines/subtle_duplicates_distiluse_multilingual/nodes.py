"""
This is a boilerplate pipeline 'subtle_duplicates_distiluse_multilingual'
generated using Kedro 0.18.7
"""

from multiprocessing import Pool, cpu_count

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from deduplication.extras.utils import (  # reduce_dimension
    find_subtle_duplicates_from_tokens,
)

model_distiluse_multilingual = SentenceTransformer(
    'distiluse-base-multilingual-cased-v2'
)


def encode_texts(texts: pd.Series) -> list:
    with Pool(int(cpu_count()/3)) as pool:
        embedded_texts = list(pool.map(
            model_distiluse_multilingual.encode,
            tqdm(texts)
        ))

    return embedded_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    concatenated_col_names: dict,
    str_cols: list,
    cols_to_be_similar: list,
    description_type: str,
    description_col: str,
    date_col: str,
    id_col: str,
    language_col: str,
    dim_tokens: int,
    threshold_similarity: float,
    threshold_semantic: float,
    threshold_partial: float,
    chunk_size: int
) -> pd.DataFrame:

    embedded_texts = encode_texts(
        data[concatenated_col_names[description_type]]
    )

    # embedded_texts = reduce_dimension(
    #     embedded_texts,
    #     dim_tokens=dim_tokens
    # )

    duplicates = find_subtle_duplicates_from_tokens(
        data,
        tokenized_texts=embedded_texts,
        str_cols=str_cols,
        cols_to_be_similar=cols_to_be_similar,
        description_col=description_col,
        date_col=date_col,
        id_col=id_col,
        language_col=language_col,
        threshold_similarity=threshold_similarity,
        threshold_semantic=threshold_semantic,
        threshold_partial=threshold_partial,
        chunk_size=chunk_size,
    )

    df_duplicates = (
        pd.DataFrame(duplicates)
        .drop_duplicates()
        .sort_values(by=["id1", "id2"], ignore_index=True)
    )
    print(
        f"{len(df_duplicates)} subtle duplicates with distiluse multilingual"
    )
    return df_duplicates
