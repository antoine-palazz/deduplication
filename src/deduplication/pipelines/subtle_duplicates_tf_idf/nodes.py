"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    find_subtle_duplicates_from_tokens
)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import warnings

tqdm.pandas()
warnings.filterwarnings('ignore')


def tokenize_tf_idf(
    texts: pd.Series,
    max_df_tokenizer: float = 0.001
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
    reduced_col_name: str = 'reduced_text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    max_df_tokenizer: float = 0.001,
    chunk_size: int = 5000,
    threshold_semantic: float = 0.95,
    threshold_partial: float = 0.1
) -> pd.DataFrame:

    tokenized_texts = tokenize_tf_idf(
        data[reduced_col_name],
        max_df_tokenizer
    )
    duplicates = find_subtle_duplicates_from_tokens(
        data,
        tokenized_texts,
        description_col,
        date_col,
        id_col,
        chunk_size,
        threshold_semantic,
        threshold_partial
    )

    df_duplicates = pd.DataFrame(
        duplicates
    ).drop_duplicates(
    ).sort_values(by=['id1', 'id2'],
                  ignore_index=True)
    print(
        f'{len(df_duplicates)} subtle duplicates were found with tf idf'
    )
    return df_duplicates
