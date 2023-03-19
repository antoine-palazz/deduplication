"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    find_subtle_duplicates_from_tokens
)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_tf_idf(
    texts: pd.Series,
    max_df_tokenizer: float
):

    vectorizer = TfidfVectorizer(
        max_df=max_df_tokenizer
    )

    tokenized_texts = vectorizer.fit_transform(texts)

    print(
        f'Vocabulary size reduced to {len(vectorizer.get_feature_names_out())}'
    )
    return tokenized_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    concatenated_col_name: str,
    str_cols: list,
    description_col: str,
    date_col: str,
    id_col: str,
    max_df_tokenizer: float,
    threshold_semantic: float,
    threshold_partial: float,
    chunk_size: int
) -> pd.DataFrame:

    tokenized_texts = tokenize_tf_idf(
        data[concatenated_col_name],
        max_df_tokenizer=max_df_tokenizer
    )
    duplicates = find_subtle_duplicates_from_tokens(
        data,
        tokenized_texts=tokenized_texts,
        str_cols=str_cols,
        description_col=description_col,
        date_col=date_col,
        id_col=id_col,
        threshold_semantic=threshold_semantic,
        threshold_partial=threshold_partial,
        chunk_size=chunk_size
    )

    df_duplicates = pd.DataFrame(
        duplicates
    ).drop_duplicates(
    ).sort_values(
        by=['id1', 'id2'],
        ignore_index=True
    )
    print(
        f'{len(df_duplicates)} subtle duplicates found with tf idf'
    )
    return df_duplicates
