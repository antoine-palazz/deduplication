"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    find_subtle_duplicates_from_tokens
)
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, logging
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device is {device}")

logging.set_verbosity_error()
tqdm.pandas()
warnings.filterwarnings('ignore')

tokenizer_multilingual_bert = BertTokenizer.from_pretrained(
    'bert-base-multilingual-uncased'
)
model_multilingual_bert = BertModel.from_pretrained(
    'bert-base-multilingual-uncased'
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
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        return torch.tensor(input_ids)


def tokenize_multilingual_bert(
    texts: pd.Series,
    batch_size: int = 64
) -> list:

    dataset = TextDataset(texts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    matrix_bert_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            outputs = model_multilingual_bert(batch)
            last_hidden_state = outputs.last_hidden_state
            matrix_bert_texts.append(
                last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
            )

    return matrix_bert_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    concatenated_col_name: str = 'text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    reduced_col_prefix: str = 'reduced_',
    batch_size: int = 64,
    chunk_size: int = 5000,
    threshold_semantic: float = 0.995,
    threshold_partial: float = 0.1
) -> pd.DataFrame:

    tokenized_texts = tokenize_multilingual_bert(
        data[reduced_col_prefix+concatenated_col_name],
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
        f'{len(df_duplicates)} subtle duplicates were found with bert'
    )
    return df_duplicates
