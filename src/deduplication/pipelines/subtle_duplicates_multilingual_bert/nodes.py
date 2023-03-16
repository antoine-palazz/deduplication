"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    find_subtle_duplicates_from_tokens
)
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, logging
import warnings

logging.set_verbosity_error()
tqdm.pandas()
warnings.filterwarnings('ignore')


def encode_text(
    text: str,
    tokenizer
):

    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    input_ids = torch.tensor(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True
        )
    ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_state = outputs.last_hidden_state
    return last_hidden_state[0][0].detach().numpy()


def tokenize_multilingual_bert(
    texts: pd.Series
) -> list:

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    with Pool(int(cpu_count()/4)) as pool:
        bert_texts = pool.map(
            partial(encode_text,
                    tokenizer=tokenizer),
            tqdm(texts)
        )

    matrix_bert_texts = [list(x) for x in bert_texts]
    return matrix_bert_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    reduced_col_name: str = 'reduced_text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    chunk_size: int = 5000,
    threshold_semantic: int = 0.995,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    tokenized_texts = tokenize_multilingual_bert(
        data[reduced_col_name]
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
        f'{len(df_duplicates)} subtle duplicates were found with bert'
    )
    return df_duplicates
