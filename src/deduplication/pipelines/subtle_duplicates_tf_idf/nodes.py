"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from deduplication.extras.utils import reduce_dimension


def tokenize_texts(
    data: pd.DataFrame,
    hyperparameters: dict,
    max_df_tokenizer: float
) -> list:

    vectorizer = TfidfVectorizer(
        max_df=max_df_tokenizer
    )

    tokenized_texts = vectorizer.fit_transform(
        data["concatenated_filtered_text"]
    )

    tokenized_texts = reduce_dimension(
        tokenized_texts,
        dim_tokens=hyperparameters["dim_tokens"]
    )

    print(
        f'Vocabulary size reduced to {len(vectorizer.get_feature_names_out())}'
    )
    return tokenized_texts
