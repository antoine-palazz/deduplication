"""
This is a boilerplate pipeline 'subtle_duplicates_distiluse_multilingual'
generated using Kedro 0.18.7
"""

from functools import partial

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import logging

from deduplication.extras.utils import reduce_dimension

logging.set_verbosity_error()
tqdm.pandas()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device for distiluse multilingual is {device}")

model_distiluse_multilingual = SentenceTransformer(
    'distiluse-base-multilingual-cased-v2'
)
model_distiluse_multilingual.to(device)


def tokenize_texts(
    data: pd.DataFrame,
    hyperparameters: dict
) -> list:
    """
    Tokenizes the offers using distiluse-base-multilingual-cased-v2

    Args:
        data (pd.DataFrame): Dataset of the offers
        hyperparameters (dict): Includes the batch size

    Returns:
        list: The list of embedded offers
    """
    embedded_texts = list(data[
        "concatenated_text"
    ].progress_apply(
        partial(model_distiluse_multilingual.encode,
                batch_size=hyperparameters["batch_size"],
                device=device
                )
    ))

    # Reduces the dimension of the embeddings to a given dimension
    embedded_texts = reduce_dimension(
        embedded_texts,
        hyperparameters=hyperparameters
    )

    return embedded_texts
