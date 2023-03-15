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
    max_df_tokenizer: float = 0.01
):

    vectorizer = TfidfVectorizer(
        max_df=max_df_tokenizer
    )

    tokenized_texts = vectorizer.fit_transform(texts)
    return tokenized_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    lemmatized_col_name: str = 'lemmatized_text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    max_df_tokenizer: float = 0.01,
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    tokenized_texts = tokenize_tf_idf(
        data[lemmatized_col_name],
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

    duplicates = pd.DataFrame(duplicates)
    print(f'{len(duplicates)} subtle duplicates were found with tf idf')
    return(duplicates)
