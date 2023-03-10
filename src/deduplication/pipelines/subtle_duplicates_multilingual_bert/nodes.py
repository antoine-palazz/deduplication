"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    create_stopwords_list,
    find_subtle_duplicates_from_tokens
)
from nltk.stem import WordNetLemmatizer
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import warnings

tqdm.pandas()
warnings.filterwarnings('ignore')


def lemmatize_texts(
    texts: pd.Series,
    stopwords_list: list
) -> pd.Series:
    lem = WordNetLemmatizer()
    lemmatized_texts = texts.progress_apply(
        lambda x: ' '.join(
            [lem.lemmatize(word) for word in x.split()
             if word not in stopwords_list
             ]
        )
    )

    return lemmatized_texts


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


def tokenize_multilingual_bert(
    lemmatized_texts: pd.Series
) -> list:

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    bert_texts = lemmatized_texts.progress_apply(
        lambda x: encode_text(x,
                              tokenizer,
                              model)
        )

    matrix_bert_texts = [list(x) for x in bert_texts]
    return matrix_bert_texts


def identify_subtle_duplicates(
    data: pd.DataFrame,
    languages_list: list,
    concatenated_col_name: str = 'text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    stopwords_list = create_stopwords_list(languages_list)
    lemmatized_texts = lemmatize_texts(
        data[concatenated_col_name],
        stopwords_list
        )
    tokenized_texts = tokenize_multilingual_bert(lemmatized_texts)

    duplicates = find_subtle_duplicates_from_tokens(
        data,
        tokenized_texts,
        languages_list,
        description_col,
        date_col,
        id_col,
        chunk_size,
        threshold_semantic,
        threshold_partial
    )

    duplicates = pd.DataFrame(duplicates)
    return(duplicates)
