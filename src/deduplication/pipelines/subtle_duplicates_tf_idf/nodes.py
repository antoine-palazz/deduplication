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
    """
    Tokenizes the offers using a TF-IDF approach

    Args:
        data (pd.DataFrame): Dataset of the offers
        hyperparameters (dict): Includes the dimension of the final tokens
        max_df_tokenizer (float): Proportion of most frequent words to remove

    Returns:
        list: List of the embedded offers
    """

    vectorizer = TfidfVectorizer(
        max_df=max_df_tokenizer
    )

    tokenized_texts = vectorizer.fit_transform(
        data["concatenated_filtered_text"]
    )

    # Reduces the dimension of the embeddings to a given dimension
    # Essential for TF-IDF as the initial embeddings have a very high dimension
    tokenized_texts = reduce_dimension(
        tokenized_texts,
        hyperparameters=hyperparameters
    )

    print(
        f'Vocabulary size reduced to {len(vectorizer.get_feature_names_out())}'
    )
    return tokenized_texts
