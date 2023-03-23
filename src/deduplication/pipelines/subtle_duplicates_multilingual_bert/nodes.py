"""
This is a boilerplate pipeline 'subtle_duplicates_multilingual_bert'
generated using Kedro 0.18.6
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, logging

from deduplication.extras.utils import (
    find_subtle_duplicates_from_tokens,
    reduce_dimension,
)

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device for multilingual BERT is {device}")

tokenizer_multilingual_bert = BertTokenizer.from_pretrained(
    "bert-base-multilingual-uncased"
)
model_multilingual_bert = BertModel.from_pretrained(
    "bert-base-multilingual-uncased"
)
model_multilingual_bert.to(device)


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        input_ids = tokenizer_multilingual_bert.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True
        )
        return torch.tensor(input_ids)


def tokenize_multilingual_bert(texts: pd.Series, batch_size: int) -> list:
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    matrix_bert_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            outputs = model_multilingual_bert(batch)
            last_hidden_state = outputs.last_hidden_state
            matrix_bert_texts.extend(
                last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
            )

    return matrix_bert_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    concatenated_col_names: dict,
    str_cols: list,
    cols_to_be_similar: list,
    description_type: str,
    description_col: str,
    date_col: str,
    id_col: str,
    dim_tokens: int,
    threshold_similarity: float,
    threshold_semantic: float,
    threshold_partial: float,
    batch_size: int,
    chunk_size: int,
) -> pd.DataFrame:

    tokenized_texts = tokenize_multilingual_bert(
        data[concatenated_col_names[description_type]],
        batch_size=batch_size
    )

    reduced_embeddings = reduce_dimension(
        tokenized_texts,
        dim_tokens=dim_tokens
    )

    duplicates = find_subtle_duplicates_from_tokens(
        data,
        tokenized_texts=reduced_embeddings,
        str_cols=str_cols,
        cols_to_be_similar=cols_to_be_similar,
        description_col=description_col,
        date_col=date_col,
        id_col=id_col,
        threshold_similarity=threshold_similarity,
        threshold_semantic=threshold_semantic,
        threshold_partial=threshold_partial,
        chunk_size=chunk_size,
    )

    df_duplicates = (
        pd.DataFrame(duplicates)
        .drop_duplicates()
        .sort_values(by=["id1", "id2"], ignore_index=True)
    )
    print(
        f"{len(df_duplicates)} subtle duplicates found with multilingual bert"
    )
    return df_duplicates
