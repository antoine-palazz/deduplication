"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from deduplication.extras.utils import reduce_dimension


def tokenize_texts(
    data: pd.DataFrame,
    concatenated_col_names: dict,
    description_type: str,
    dim_tokens: int,
    max_df_tokenizer: float
) -> list:

    vectorizer = TfidfVectorizer(
        max_df=max_df_tokenizer
    )

    tokenized_texts = vectorizer.fit_transform(
        data[concatenated_col_names[description_type]]
    )

    tokenized_texts = reduce_dimension(
        tokenized_texts,
        dim_tokens=dim_tokens
    )

    print(
        f'Vocabulary size reduced to {len(vectorizer.get_feature_names_out())}'
    )
    return tokenized_texts
