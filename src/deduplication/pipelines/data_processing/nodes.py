"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import re
from tqdm import tqdm
from unidecode import unidecode
import warnings

nltk.download('stopwords')
tqdm.pandas()
warnings.filterwarnings('ignore')


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna("")
    return data


def normalize_strings(
    data: pd.DataFrame,
    str_cols: list
) -> pd.DataFrame:

    data[str_cols] = data[str_cols].progress_apply(
        lambda x: x.str.replace(
            '\n\r', ' ').replace(
                '\n', ' ').replace(
                    r'\W', ' ').apply(
            lambda x: unidecode(re.sub(' +', ' ', x))
        ).str.strip().str.lower()
    )
    return data


def create_concatenated_column(
    data: pd.DataFrame,
    str_cols: list,
    concatenated_col_name: str
) -> pd.DataFrame:

    data[concatenated_col_name] = data[str_cols[0]]
    for col in str_cols[1:]:
        data[concatenated_col_name] += ' ' + data[col]

    return data


def preprocess_data(
    data: pd.DataFrame,
    str_cols: list = ['title', 'company_name', 'location', 'description'],
    concatenated_col_name: str = 'text'
) -> pd.DataFrame:

    data = remove_nans(data)
    data = normalize_strings(data, str_cols)
    data = create_concatenated_column(data, str_cols, concatenated_col_name)
    print(f'{len(data)} ads in the preprocessed file')

    return data


def create_stopwords_list(languages_list: list) -> list:
    stopwords_list = stopwords.words(languages_list)
    return stopwords_list


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


def create_lemmatized_col(
    data: pd.DataFrame,
    languages_list: list,
    concatenated_col_name: str = 'text',
    lemmatized_col_name: str = 'lemmatized_text',
) -> pd.DataFrame:

    stopwords_list = create_stopwords_list(languages_list)
    data[lemmatized_col_name] = lemmatize_texts(
        data[concatenated_col_name],
        stopwords_list
        )

    return data
