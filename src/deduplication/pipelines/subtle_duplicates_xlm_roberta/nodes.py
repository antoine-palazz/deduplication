"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    find_subtle_duplicates_from_tokens
)
import pandas as pd
import torch
from tqdm import tqdm
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import warnings

tqdm.pandas()
warnings.filterwarnings('ignore')


def encode_text(
    text: str,
    tokenizer,
    model
):
    input_ids = torch.tensor(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True
        )
    ).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state[0][0].detach().numpy()


def tokenize_xlm_roberta(
    texts: pd.Series
) -> list:

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

    roberta_texts = texts.progress_apply(
        lambda x: encode_text(x,
                              tokenizer,
                              model)
        )

    matrix_roberta_texts = [list(x) for x in roberta_texts]
    return matrix_roberta_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    lemmatized_col_name: str = 'lemmatized_text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    tokenized_texts = tokenize_xlm_roberta(data[lemmatized_col_name])

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
    return(duplicates)
