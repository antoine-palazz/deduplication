"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from functools import partial
import html
from multiprocessing import Pool, cpu_count
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
    beginning_prefix: str,
    end_prefix: str,
    threshold_short_description: int = 200
) -> pd.DataFrame:

    data_with_new_cols = data.copy()
    data_with_new_cols[concatenated_col_name] = data[cols_to_concatenate[0]]
    for col in tqdm(cols_to_concatenate[1:]):
        data_with_new_cols[concatenated_col_name] += ' ' + data[col]

    # Also throw shorts description in the lot
    data_with_new_cols[beginning_prefix+description_col] = data_with_new_cols[
        description_col
    ].apply(lambda x: x[:threshold_short_description])
    data_with_new_cols[end_prefix+description_col] = data_with_new_cols[
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
    beginning_prefix: str = 'beginning_',
    end_prefix: str = 'end_',
    threshold_short_description: int = 200
) -> pd.DataFrame:

    data_1 = remove_nans(data)
    data_2 = remove_html(data_1, str_cols)
    data_3 = normalize_strings(data_2, str_cols)
    data_4 = create_concatenated_column(data_3,
                                        cols_to_concatenate,
                                        concatenated_col_name,
                                        description_col,
                                        beginning_prefix,
                                        end_prefix,
                                        threshold_short_description)
    print(f'{len(data_4)} ads in the preprocessed file')

    return data_4.sort_values(by='id')


def create_stopwords_list(languages_list: list) -> set:
    stopwords_list = set(stopwords.words(languages_list))
    return stopwords_list


def remove_stopwords(
    texts: pd.Series,
    stopwords_list: set
) -> pd.Series:

    texts_no_stopwords = texts.progress_apply(
        lambda x: ' '.join(
            [word for word in x.split()
             if word not in stopwords_list or
                len(word) < 2
             ]
        )
    )

    return texts_no_stopwords


def lemmatize_text(
    text: str,
    lemmatizer
) -> str:

    lemmatized_text = ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split()]
    )

    return lemmatized_text


def lemmatize_texts(
    texts: pd.Series,
) -> pd.Series:

    lem = WordNetLemmatizer()

    with Pool(int(cpu_count()/3)) as pool:
        lemmatized_texts = pool.map(
            partial(lemmatize_text, lemmatizer=lem),
            tqdm(texts)
        )

    return pd.Series(lemmatized_texts)


def filter_out_too_frequent_words(
    texts: pd.Series,
    proportion_words_to_filter_out: float = 0.50,
) -> pd.Series:

    corpus = " ".join(texts)
    tokens = nltk.word_tokenize(corpus)
    frequencies = nltk.FreqDist(tokens)

    vocabulary_size = len(set(tokens))
    n_to_filter_out = int(proportion_words_to_filter_out * vocabulary_size)

    most_common_words = [
        word for word, freq in frequencies.most_common(n_to_filter_out)
    ]
    filtered_texts = remove_stopwords(texts, most_common_words)

    return filtered_texts


def create_reduced_text_cols(
    data: pd.DataFrame,
    languages_list: list,
    concatenated_col_name: str = 'text',
    reduced_col_prefix: str = 'reduced_',
    very_reduced_col_prefix: str = 'very_reduced_',
    proportion_words_to_filter_out: float = 0.50
) -> pd.DataFrame:

    stopwords_list = create_stopwords_list(languages_list)
    reduced_texts = data[concatenated_col_name]
    data_with_reduced_texts = data.copy()

    reduced_texts_1 = remove_stopwords(
        reduced_texts,
        stopwords_list
    )
    reduced_texts_2 = lemmatize_texts(
        reduced_texts_1
    )

    reduced_texts_3 = filter_out_too_frequent_words(
        reduced_texts_2,
        proportion_words_to_filter_out=proportion_words_to_filter_out
    )

    data_with_reduced_texts[
        reduced_col_prefix + concatenated_col_name
    ] = reduced_texts_2
    data_with_reduced_texts[
        very_reduced_col_prefix + concatenated_col_name
    ] = reduced_texts_3

    return data_with_reduced_texts
