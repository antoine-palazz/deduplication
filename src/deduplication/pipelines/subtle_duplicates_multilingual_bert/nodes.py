"""
This is a boilerplate pipeline 'subtle_duplicates_multilingual_bert'
generated using Kedro 0.18.6
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, logging

from deduplication.extras.utils import reduce_dimension

logging.set_verbosity_error()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"The device for multilingual BERT is {device}")

tokenizer_multilingual_bert = BertTokenizer.from_pretrained(
    "bert-base-multilingual-uncased"
)
model_multilingual_bert = BertModel.from_pretrained(
    "bert-base-multilingual-uncased"
)
# model_multilingual_bert.to(device)


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


def tokenize_texts(
    data: pd.DataFrame,
    concatenated_col_names: dict,
    description_type: str,
    dim_tokens: int,
    batch_size: int
) -> list:

    dataset = TextDataset(
        data[concatenated_col_names[description_type]]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    matrix_bert_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # batch = batch.to(device)
            outputs = model_multilingual_bert(batch)
            last_hidden_state = outputs.last_hidden_state
            matrix_bert_texts.extend(
                last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
            )

    matrix_bert_texts = reduce_dimension(
        matrix_bert_texts,
        dim_tokens=dim_tokens
    )

    return matrix_bert_texts
