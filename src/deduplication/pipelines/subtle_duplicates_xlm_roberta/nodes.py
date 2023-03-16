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
    texts: pd.Series,
    batch_size: int = 1
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


def tokenize_xlm_roberta_by_batch(
    texts: pd.Series,
    batch_size: int = 128
) -> list:

    n_ads = len(texts)
    matrix_bert_texts = []
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

    for i in tqdm(range(0, n_ads, batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_input_ids = []
        for text in batch_texts:
            input_ids = tokenizer.encode(text,
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding='max_length')
            batch_input_ids.append(input_ids)
        batch_input_ids = torch.tensor(batch_input_ids)
        with torch.no_grad():
            outputs = model(batch_input_ids)
            last_hidden_states = outputs.last_hidden_state
        matrix_bert_texts.extend(
            [list(x[0]) for x in last_hidden_states.detach().numpy()]
        )

    return matrix_bert_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    reduced_col_name: str = 'reduced_text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    batch_size: int = 128,
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    tokenized_texts = tokenize_xlm_roberta_by_batch(
        data[reduced_col_name],
        batch_size=batch_size
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
        f'{len(df_duplicates)} subtle duplicates were found with xlm roberta'
    )
    return(df_duplicates)
