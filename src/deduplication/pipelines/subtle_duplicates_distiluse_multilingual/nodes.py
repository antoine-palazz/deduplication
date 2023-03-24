"""
This is a boilerplate pipeline 'subtle_duplicates_distiluse_multilingual'
generated using Kedro 0.18.7
"""

from functools import partial

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from deduplication.extras.utils import reduce_dimension

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device for distiluse multilingual is {device}")

model_distiluse_multilingual = SentenceTransformer(
    'distiluse-base-multilingual-cased-v2'
)
model_distiluse_multilingual.to(device)


def tokenize_texts(
    data: pd.DataFrame,
    concatenated_col_names: dict,
    description_type: str,
    dim_tokens: int,
    batch_size: int
) -> list:

    embedded_texts = list(data[
        concatenated_col_names[description_type]
    ].progress_apply(
        partial(model_distiluse_multilingual.encode,
                batch_size=batch_size,
                device=device
                )
    ))

    embedded_texts = reduce_dimension(
        embedded_texts,
        dim_tokens=dim_tokens
    )

    return embedded_texts
