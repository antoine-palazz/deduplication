"""
This is a boilerplate pipeline 'subtle_duplicates_xlm_roberta'
generated using Kedro 0.18.6
"""

import nltk
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging

from deduplication.extras.utils import reduce_dimension

# torch.cuda.empty_cache()
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logging.set_verbosity_error()

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"The device for XLM Roberta is {device}")

tokenizer_xlm_roberta = AutoTokenizer.from_pretrained("xlm-roberta-base")
model_xlm_roberta = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
# model_xlm_roberta.to(device)


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

        corpus = " ".join(texts)  # All the concatenated offers in one string
        tokens = nltk.word_tokenize(corpus)
        vocab_size = len(set(tokens))
        print(f'The vocab size for XLM Roberta is {vocab_size}')

        # Re-train ROBERTA with our corpus to specialize it
        self.tokenizer = tokenizer_xlm_roberta.train_new_from_iterator(
            split(texts, 64),
            vocab_size=vocab_size
        )
        print('XLM Roberta tokenizer is re-trained')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        input_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True
        )
        return torch.tensor(input_ids)


def split(a: list, n: int) -> list:
    """
    Splits a list `a` into `n` roughly equal-sized sub-lists.

    Args:
        a (list): The list to split
        n (int): The number of sub-lists to create

    Returns:
        list: A generator that produces `n` sub-lists of `a`
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def tokenize_texts(
    data: pd.DataFrame,
    hyperparameters: dict
) -> list:
    """
    Tokenizes the offers using a re-trained version of xlm-roberta-base
    specialized on our corpus

    Args:
        data (pd.DataFrame): Dataset of offers
        hyperparameters (dict): Includes the batch size

    Returns:
        list: List of embedded offers
    """

    dataset = TextDataset(data["concatenated_text"])
    dataloader = DataLoader(dataset, batch_size=hyperparameters["batch_size"])

    matrix_roberta_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # batch = batch.to(device)
            outputs = model_xlm_roberta(batch,
                                        output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            matrix_roberta_texts.extend(
                last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
            )

    # Reduces the dimension of the embeddings to a given dimension
    matrix_roberta_texts = reduce_dimension(
        matrix_roberta_texts,
        hyperparameters=hyperparameters
    )

    print(
        f"Tokens matrix:\
            {len(matrix_roberta_texts)} x {len(matrix_roberta_texts[0])}"
    )
    return matrix_roberta_texts
