"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

import html
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import re
from tqdm import tqdm
from unidecode import unidecode
import warnings

nltk.download('stopwords')
nltk.download('wordnet')
tqdm.pandas()
warnings.filterwarnings('ignore')


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    data_without_nans = data.fillna("")
    return data_without_nans


def remove_html(
    data: pd.DataFrame,
    str_cols: list
) -> pd.DataFrame:

    data_without_html = data.copy()
    data_without_html[str_cols] = data[str_cols].progress_applymap(
        lambda x: re.sub('<[^<]+?>', " ", html.unescape(x))
    )

    return data_without_html


def normalize_strings(  # To improve, for instance with balises
    data: pd.DataFrame,
    str_cols: list
) -> pd.DataFrame:

    clean_data = data.copy()
    clean_data[str_cols] = data[str_cols].progress_apply(
        lambda x: x.str.replace(
            r'\r|\n', ' ', regex=True
        ).replace(
            r'\W', ' ', regex=True
        ).replace(
            r' +', ' ', regex=True
        ).apply(unidecode).str.lower().str.strip()
    )
    return clean_data


def create_concatenated_column(
    data: pd.DataFrame,
    cols_to_concatenate: list,
    concatenated_col_name: str,
    description_col: str,
    threshold_short_description: int = 200
) -> pd.DataFrame:

    data_with_new_cols = data.copy()
    data_with_new_cols[concatenated_col_name] = data[cols_to_concatenate[0]]
    for col in tqdm(cols_to_concatenate[1:]):
        data_with_new_cols[concatenated_col_name] += ' ' + data[col]

    # Also throw shorts description in the lot
    data_with_new_cols['beginning_description'] = data_with_new_cols[
        description_col
    ].apply(lambda x: x[:threshold_short_description])
    data_with_new_cols['end_description'] = data_with_new_cols[
        description_col
    ].apply(lambda x: x[-threshold_short_description:])

    return data_with_new_cols


def preprocess_data(
    data: pd.DataFrame,
    str_cols: list = ['title', 'company_name', 'location', 'description'],
    cols_to_concatenate: list = ['title',
                                 'company_name',
                                 'location',
                                 'country_id',
                                 'description'],
    concatenated_col_name:  str = 'text',
    description_col: str = 'description',
    threshold_short_description: int = 200
) -> pd.DataFrame:

    data_1 = remove_nans(data)
    data_2 = remove_html(data_1, str_cols)
    data_3 = normalize_strings(data_2, str_cols)
    data_4 = create_concatenated_column(data_3,
                                        cols_to_concatenate,
                                        concatenated_col_name,
                                        description_col,
                                        threshold_short_description)
    print(f'{len(data_4)} ads in the preprocessed file')

    return data_4


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
    data_with_lemmas = data.copy()
    data_with_lemmas[lemmatized_col_name] = lemmatize_texts(
        data[concatenated_col_name],
        stopwords_list
        )

    return data_with_lemmas
