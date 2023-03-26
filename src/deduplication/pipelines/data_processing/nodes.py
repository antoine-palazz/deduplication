"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

import html
import re
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import nltk
import pandas as pd
from ftlangdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from unidecode import unidecode

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tqdm.pandas()
warnings.filterwarnings('ignore')


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    data_without_nans = data.fillna("")
    return data_without_nans


def remove_html(
    texts: pd.Series
) -> pd.Series:

    texts_without_html = texts.apply(
        lambda x: re.sub('<[^<]+?>', " ", html.unescape(x))
    ).str.replace(
        r'\r|\n', ' ', regex=True
    ).replace(
        r' +', ' ', regex=True
    ).str.strip()

    return texts_without_html


def normalize_strings(
    texts: pd.Series
) -> pd.Series:

    clean_texts = texts.str.lower()

    return clean_texts


def find_language_from_text(
    text: str
) -> str:
    text = detect(text)['lang']
    return text


def preprocess_data_basic(
    data: pd.DataFrame,
    str_cols: list,
    description_col: str,
    id_col: str,
    language_col: str
) -> pd.DataFrame:

    preprocessed_data = remove_nans(data)

    preprocessed_data[str_cols] = preprocessed_data[str_cols].progress_apply(
        remove_html
    )

    preprocessed_data[str_cols] = preprocessed_data[str_cols].progress_apply(
        normalize_strings
    )

    preprocessed_data[language_col] = preprocessed_data[
        description_col
    ].progress_apply(find_language_from_text)

    return preprocessed_data.sort_values(by=id_col)


def filter_out_incomplete_offers(
    data: pd.DataFrame,
    required_cols: list,
    nb_allowed_nans: int
) -> pd.DataFrame:

    filtered_data_on_nans = data[
        data.apply(lambda x: x == "").sum(axis=1) <= nb_allowed_nans
    ]

    filtered_data_on_cols = filtered_data_on_nans[
        (
            filtered_data_on_nans[required_cols].apply(lambda x: x != "")
        ).all(axis=1)
    ]

    filtered_data_on_cols.reset_index(
        drop=True,
        inplace=True
    )

    print(f'Length of the filtered table: {len(filtered_data_on_cols)}')
    return filtered_data_on_cols


def remove_special_characters(
    texts: pd.Series
) -> pd.DataFrame:

    clean_texts = texts.str.replace(
        r'[\W\d]+', ' ', regex=True
    ).replace(
        r' +', ' ', regex=True
    ).apply(unidecode).str.strip()

    return clean_texts


def create_stopwords_list(
    languages_list: list,
    without_accents: bool
) -> set:
    stopwords_list = stopwords.words(languages_list)
    lowered_stopwords_list = map(
        lambda x: x.lower().strip(),
        stopwords_list
    )
    if without_accents:
        standardized_stopwords_list = map(
            unidecode,
            lowered_stopwords_list
        )
    final_stopwords_set = sorted(set(standardized_stopwords_list))
    return final_stopwords_set


def remove_stopwords_from_text(
    text: str,
    stopwords_list: set
) -> str:

    text_no_stopwords = ' '.join(
        [word for word in text.split()
            if word not in stopwords_list and
            len(word) >= 2
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
            texts
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


def filter_out_words_in_one_language(
    texts: pd.Series,
    proportion_words_to_filter_out: float
) -> pd.Series:

    corpus = " ".join(texts)
    tokens = nltk.word_tokenize(corpus)
    frequencies = nltk.FreqDist(tokens)

    vocab_size = len(set(tokens))
    n_to_filter_out = int(proportion_words_to_filter_out * vocab_size)

    most_common_words = sorted(set([
        word for word, freq in frequencies.most_common(n_to_filter_out)
    ]))
    filtered_texts = remove_stopwords(
        texts,
        stopwords_list=most_common_words
    )

    return filtered_texts


def filter_out_too_frequent_words(
    data: pd.DataFrame,
    description_col: str,
    language_col: str,
    proportion_words_to_filter_out: float
) -> pd.DataFrame:

    languages_list = set(data[language_col])
    well_described_data = data.copy()

    for language in tqdm(languages_list):

        data_lang = data[data[language_col] == language]
        data_lang_idxs = data_lang.index

        filtered_descriptions_lang = filter_out_words_in_one_language(
            data_lang[description_col],
            proportion_words_to_filter_out=proportion_words_to_filter_out
        )

        well_described_data.loc[
            data_lang_idxs, description_col
        ] = filtered_descriptions_lang

    print(
        f'Proportion of words removed: {proportion_words_to_filter_out}'
        )
    return well_described_data


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

    data_with_new_cols[
        concatenated_col_name
    ] = data_with_new_cols[
        concatenated_col_name
    ].replace(
        r' +', ' ', regex=True
    ).str.strip()

    return data_with_new_cols


def preprocess_data_extensive(
    pre_preprocessed_data: pd.DataFrame,
    str_cols: list,
    cols_to_concatenate: dict,
    description_col: str,
    filtered_description_col: str,
    language_col: str,
    concatenated_col_names: dict,
    languages_list: list,
    beginning_prefix: str,
    end_prefix: str,
    proportion_words_to_filter_out: float,
    threshold_short_text: int
) -> pd.DataFrame:

    preprocessed_data = pre_preprocessed_data.copy()
    stopwords_list = create_stopwords_list(
        languages_list,
        without_accents=True
    )

    preprocessed_data[str_cols] = preprocessed_data[str_cols].progress_apply(
        remove_special_characters
    )

    preprocessed_data[filtered_description_col] = remove_stopwords(
        preprocessed_data[description_col],
        stopwords_list=stopwords_list
    )

    preprocessed_data[filtered_description_col] = lemmatize_texts(
        preprocessed_data[filtered_description_col]
    )

    preprocessed_data = filter_out_too_frequent_words(
        preprocessed_data,
        description_col=filtered_description_col,
        language_col=language_col,
        proportion_words_to_filter_out=proportion_words_to_filter_out
    )

    preprocessed_data = create_extra_cols_from_text_cols(
        preprocessed_data,
        cols_to_duplicate=[
            description_col,
            filtered_description_col
            ],
        beginning_prefix=beginning_prefix,
        end_prefix=end_prefix,
        threshold_short_text=threshold_short_text
    )

    preprocessed_data = create_concatenated_column(
        preprocessed_data,
        cols_to_concatenate=cols_to_concatenate['normal'],
        concatenated_col_name=concatenated_col_names['normal']
    )

    preprocessed_data = create_concatenated_column(
        preprocessed_data,
        cols_to_concatenate=cols_to_concatenate['filtered'],
        concatenated_col_name=concatenated_col_names['filtered']
    )

    return preprocessed_data


def filter_international_companies(
    preprocessed_data: pd.DataFrame,
    cols_to_be_diversified: list,
    company_col: str
) -> pd.DataFrame:

    # Count unique values in each of the columns of interest for each company
    unique_counts = preprocessed_data.groupby(
        [company_col]
    )[cols_to_be_diversified].nunique()

    # Filter out companies with only one unique value in each of them
    international_offers = preprocessed_data[
        preprocessed_data[company_col].isin(
            unique_counts[(unique_counts > 1).any(axis=1)].index
        )
    ].reset_index(drop=True)

    print(
        f'Nb of offers by international companies: {len(international_offers)}'
    )
    return international_offers


def filter_out_poorly_described_offers(
    preprocessed_data: pd.DataFrame,
    cols_not_to_be_diversified: list,
    description_col: str
) -> pd.DataFrame:

    data_with_one_col_not_to_be_diversified = create_concatenated_column(
        preprocessed_data,
        cols_to_concatenate=cols_not_to_be_diversified,
        concatenated_col_name="col_not_to_be_diversified"
    )

    # Count unique values for tuple of cols of interest for each description
    unique_counts = data_with_one_col_not_to_be_diversified.groupby(
        [description_col]
    )[["col_not_to_be_diversified"]].nunique()

    # Filter out companies with several unique values in any of them
    well_described_offers = preprocessed_data[
        preprocessed_data[description_col].isin(
            unique_counts[(unique_counts == 1).any(axis=1)].index
        )
    ].reset_index(drop=True)

    print(
        f'Nb of not poorly described offers: {len(well_described_offers)}'
    )
    return well_described_offers
