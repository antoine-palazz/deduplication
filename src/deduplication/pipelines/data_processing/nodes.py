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

nltk.download('punkt')
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


def normalize_strings(
    data: pd.DataFrame,
    str_cols: list
) -> pd.DataFrame:

    clean_data = data.copy()
    clean_data[str_cols] = data[str_cols].progress_apply(
        lambda x: x.str.replace(
            r'\r|\n', ' ', regex=True
        ).replace(
            r' +', ' ', regex=True
        ).str.lower().str.strip()
    )
    return clean_data


def preprocess_data_basic(
    data: pd.DataFrame,
    str_cols: list,
) -> pd.DataFrame:

    data_1 = remove_nans(data)
    data_2 = remove_html(data_1, str_cols)
    data_3 = normalize_strings(data_2, str_cols)

    return data_3.sort_values(by='id')


def filter_out_incomplete_offers(
    data: pd.DataFrame,
    required_cols: list,
    nb_allowed_nans: int,

) -> pd.DataFrame:

    filtered_data_on_nans = data[
        data.apply(lambda x: x == "").sum(axis=1) <= nb_allowed_nans
    ]

    filtered_data_on_cols = filtered_data_on_nans[
        (filtered_data_on_nans[required_cols] != "").all()
    ]

    return filtered_data_on_cols


def remove_special_characters(
    texts: pd.Series
) -> pd.DataFrame:

    clean_texts = texts.str.replace(
        r'\W', ' ', regex=True
    ).replace(
        r' +', ' ', regex=True
    ).apply(unidecode).str.strip()

    return clean_texts


def create_stopwords_list(
    languages_list: list,
    without_accents: bool
) -> set:
    stopwords_list = stopwords.words(languages_list)
    if without_accents:
        stopwords_list = map(unidecode, stopwords_list)
    stopwords_set = sorted(set(stopwords_list))
    return stopwords_set


def remove_stopwords_from_text(
    text: str,
    stopwords_list: set
) -> str:

    text_no_stopwords = ' '.join(
        [word for word in text.split()
            if word not in stopwords_list or
            len(word) < 2
         ]
    )

    return text_no_stopwords


def remove_stopwords(
    texts: pd.Series,
    stopwords_list: set
) -> pd.Series:

    with Pool(int(cpu_count()/3)) as pool:
        texts_no_stopwords = pool.map(
            partial(
                remove_stopwords_from_text,
                stopwords_list=stopwords_list),
            tqdm(texts)
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
    proportion_words_to_filter_out: float,
) -> pd.Series:

    corpus = " ".join(texts)
    tokens = nltk.word_tokenize(corpus)
    frequencies = nltk.FreqDist(tokens)

    vocab_size = len(set(tokens))
    n_to_filter_out = int(proportion_words_to_filter_out * vocab_size)
    print(
        f'{n_to_filter_out} too commons words out of {vocab_size} filtered out'
    )

    most_common_words = sorted(set([
        word for word, freq in frequencies.most_common(n_to_filter_out)
    ]))
    filtered_texts = remove_stopwords(
        texts,
        stopwords_list=most_common_words
    )

    return filtered_texts


def create_extra_cols_from_text_cols(
    data: pd.DataFrame,
    cols_to_duplicate: list,
    beginning_prefix: str,
    end_prefix: str,
    threshold_short_text: int
) -> pd.DataFrame:

    data_with_new_cols = data.copy()

    for col in cols_to_duplicate:

        data_with_new_cols[beginning_prefix+col] = data_with_new_cols[
            col
        ].apply(lambda x: x[:threshold_short_text])
        data_with_new_cols[end_prefix+col] = data_with_new_cols[
            col
        ].apply(lambda x: x[-threshold_short_text:])

    return data_with_new_cols


def create_concatenated_column(
    data: pd.DataFrame,
    cols_to_concatenate: list,
    concatenated_col_name: str
) -> pd.DataFrame:

    data_with_new_cols = data.copy()
    data_with_new_cols[concatenated_col_name] = data[cols_to_concatenate[0]]
    for col in tqdm(cols_to_concatenate[1:]):
        data_with_new_cols[concatenated_col_name] += ' ' + data[col]

    return data_with_new_cols


def preprocess_data_extensive(
    preprocessed_data: pd.DataFrame,
    str_cols: list,
    description_col: list,
    cols_to_concatenate: list,
    concatenated_col_name: str,
    languages_list: list,
    beginning_prefix: str,
    end_prefix: str,
    proportion_words_to_filter_out: float,
    threshold_short_text: int
) -> pd.DataFrame:

    preprocessed_data = preprocessed_data.copy()
    stopwords_list = create_stopwords_list(
        languages_list,
        without_accents=True
    )

    preprocessed_data[str_cols] = preprocessed_data[str_cols].progress_apply(
        remove_special_characters
    )

    preprocessed_data[str_cols] = preprocessed_data[str_cols].progress_apply(
        partial(
            remove_stopwords,
            stopwords_list=stopwords_list
        )
    )

    preprocessed_data[str_cols] = preprocessed_data[str_cols].progress_apply(
        lemmatize_texts
    )

    preprocessed_data[description_col] = filter_out_too_frequent_words(
        preprocessed_data[description_col],
        proportion_words_to_filter_out=proportion_words_to_filter_out
    )

    preprocessed_data = create_extra_cols_from_text_cols(
        preprocessed_data,
        cols_to_duplicate=[description_col],
        beginning_prefix=beginning_prefix,
        end_prefix=end_prefix,
        threshold_short_text=threshold_short_text
    )

    preprocessed_data = create_concatenated_column(
        preprocessed_data,
        cols_to_concatenate=cols_to_concatenate,
        concatenated_col_name=concatenated_col_name
    )

    return preprocessed_data
