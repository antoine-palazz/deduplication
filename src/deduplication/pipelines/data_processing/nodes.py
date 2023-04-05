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
import torch
from ftlangdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    logging,
    pipeline,
)
from unidecode import unidecode

logging.set_verbosity_error()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tqdm.pandas()
warnings.filterwarnings('ignore')

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The device for NER is {device}")

# If we want to cumpute NER:
tokenizer_ner = AutoTokenizer.from_pretrained(
    "Davlan/distilbert-base-multilingual-cased-ner-hrl"
)
model_ner = AutoModelForTokenClassification.from_pretrained(
    "Davlan/distilbert-base-multilingual-cased-ner-hrl"
)
model_ner.to(device)
ner_pipeline = pipeline(
    "ner",
    model=model_ner,
    tokenizer=tokenizer_ner,
    device=device
)


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces all the NaNs in data by empty strings

    Args:
        data (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    data_without_nans = data.fillna("")
    return data_without_nans


def remove_html(
    texts: pd.Series
) -> pd.Series:
    """
    Corrects some issues with the strings related to their retrieval:
    - HTML tags, "*" for anonymizations, etc.

    Args:
        texts (pd.Series): Vector of texts

    Returns:
        pd.Series: "Corrected" vector of texts
    """
    texts_without_html = texts.apply(
        lambda x: re.sub('<[^<]+?>', " ", html.unescape(x))
    ).str.replace(
        r'\r|\n', ' ', regex=True
    ).str.replace(
        '*', ' '
    ).str.replace(
        "&#39;", "'"
    ).replace(
        r' +', ' ', regex=True
    ).str.strip()

    return texts_without_html


def normalize_strings(
    texts: pd.Series
) -> pd.Series:
    """
    Lowers a vector of texts

    Args:
        texts (pd.Series): Vector of texts

    Returns:
        pd.Series: Lowered texts
    """
    clean_texts = texts.str.lower()

    return clean_texts


def find_language_from_text(
    text: str
) -> str:
    """
    Uses Fasttext to detect the language from a text.
    The reliability is not always very good, but for our use
    it is not very important.

    Args:
        text (str)

    Returns:
        str: Guessed language of the text
    """
    text = detect(text)['lang']
    return text


def preprocess_data_very_basic(
    data: pd.DataFrame,
    str_cols: dict,
    ner: dict
) -> pd.DataFrame:
    """
    Performs very slight preprocessing on the initial dataset
    (just enough not to disturb the NER algorithm):
    - Removes NaNs and web undesirable content in strings
    - Creates an aggregated column
    - Computes NER if asked to

    Args:
        data (pd.DataFrame): Initial dataset
        str_cols (dict): Columns that will be furhter compared
        ner (dict): Booleans to know if we want to use NER

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    preprocessed_data = remove_nans(data)

    preprocessed_data[str_cols["normal"]] = preprocessed_data[
        str_cols["normal"]
    ].progress_apply(
        remove_html
    )

    preprocessed_data = create_concatenated_column(
        data=preprocessed_data,
        list_cols_to_concatenate=str_cols["normal"],
        concatenated_col_name="concatenated_raw_text"
    )

    if ner['compute']:
        # Very long to run + requires GPU
        # Only useful for partials identification in utils.py
        preprocessed_data = encode_ner(preprocessed_data)

    return preprocessed_data


def preprocess_data_basic(
    data: pd.DataFrame,
    str_cols: dict,
    ner: dict
) -> pd.DataFrame:
    """
    Performs slight preprocessing on the initial dataset
    (just enough not to add false full duplicates):
    - Performs the very slight preprocessing above
    - Lowers the texts
    - Finds the languages of the descriptions

    Args:
        data (pd.DataFrame): Initial dataset
        str_cols (dict): Columns that will be furhter compared
        ner (dict): Booleans to know if we want to use NER

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    preprocessed_data = preprocess_data_very_basic(
        data,
        str_cols=str_cols,
        ner=ner)

    preprocessed_data[str_cols["normal"]] = preprocessed_data[
        str_cols["normal"]
    ].progress_apply(
        normalize_strings
    )

    preprocessed_data["language"] = preprocessed_data[
        "description"
    ].progress_apply(find_language_from_text)

    return preprocessed_data.sort_values(by="id")


def filter_out_incomplete_offers(
    data: pd.DataFrame,
    type_easy: str,
    required_cols_for_filtering: dict,
    nb_allowed_nans_for_filtering: dict,
) -> pd.DataFrame:
    """
    Filters the offers based on given conditions:
    - How many missing fields do we allow?
    - Are there fields that we don't want to be NaN?

    Args:
        data (pd.DataFrame): Dataset of offers
        type_easy (str): Which "easy" duplicates are we looking for?
        required_cols_for_filtering (dict): Which cols cannot be NaN?
        nb_allowed_nans_for_filtering (dict): How many NaNs allowed?

    Returns:
        pd.DataFrame: Filtered dataset of offers
    """
    if "NER" in data.columns:
        data_no_ner = data.drop("NER", axis=1)
    else:
        data_no_ner = data.copy()

    filtered_data_on_nans = data[
        (data_no_ner.apply(lambda x: x == "").sum(axis=1) <=
         nb_allowed_nans_for_filtering[type_easy])
    ]  # Removes rows with too many NaNs

    filtered_data_on_cols = filtered_data_on_nans[
        (
            filtered_data_on_nans[
                required_cols_for_filtering[type_easy]
            ].apply(lambda x: x != "")
        ).all(axis=1)
    ]  # Remove NaNs with missing essential columns

    filtered_data_on_cols.reset_index(
        drop=True,
        inplace=True
    )

    print(f'Length of the filtered table: {len(filtered_data_on_cols)}')
    return filtered_data_on_cols


def remove_special_characters(
    texts: pd.Series
) -> pd.DataFrame:
    """
    Removes all special characters from a vector of texts
    Remain only alphabetic characters without accents

    Args:
        texts (pd.Series): Vector of texts

    Returns:
        pd.DataFrame: "Cleaned" vector of texts
    """
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
    """
    Creates a list of stopwords for a set of given languages

    Args:
        languages_list (list): Languages from which to take stopwords
        without_accents (bool): Do we remove accents from stopwords?

    Returns:
        set: Set of multilingual stopwords
    """
    stopwords_list = stopwords.words(languages_list)
    stopwords_list = map(
        lambda x: x.lower().strip(),
        stopwords_list
    )
    if without_accents:  # If we want no accents in the stopwords
        stopwords_list = map(
            unidecode,
            stopwords_list
        )
    final_stopwords_set = sorted(set(stopwords_list))
    return final_stopwords_set


def remove_stopwords_from_text(
    text: str,
    stopwords_list: set
) -> str:
    """
    Removes stopwords and words with less than 3 letters from a text

    Args:
        text (str)
        stopwords_list (set)

    Returns:
        str: Text without the stopwords
    """
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
    """
    Removes stopwords from a vector of texts using multiprocessing

    Args:
        texts (pd.Series): vector of texts
        stopwords_list (set)

    Returns:
        pd.Series: "Cleaned" vector of texts
    """
    with Pool(int(cpu_count()/2)) as pool:  # Multiprocessing
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
    """
    Lemmatizes a given text using WordNetLemmatizer
    Not very efficient: considers all words as English nouns
    But not very important for our usage
    Caution: do not use before a Transformer

    Args:
        text (str)
        lemmatizer (_type_): for now WordNetLemmatizer

    Returns:
        str: Lemmatized text
    """
    lemmatized_text = ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split()]
    )

    return lemmatized_text


def lemmatize_texts(
    texts: pd.Series,
) -> pd.Series:
    """
    Lemmatizes vector of texts using multiprocessing

    Args:
        texts (pd.Series): vector of texts

    Returns:
        pd.Series: Lemmatized vector of texts
    """
    lem = WordNetLemmatizer()  # Is there a better one to use?

    with Pool(int(cpu_count()/2)) as pool:  # Multiprocessing
        lemmatized_texts = pool.map(
            partial(lemmatize_text, lemmatizer=lem),
            tqdm(texts)
        )

    return pd.Series(lemmatized_texts)


def filter_out_words_in_one_language(
    texts: pd.Series,
    proportion_words_to_filter_out: float
) -> pd.Series:
    """
    Removes a proportion from the most frequent words of a vector of texts

    Args:
        texts (pd.Series): Vector of texts
        proportion_words_to_filter_out (float)

    Returns:
        pd.Series: Vector of texts without these most frequent words
    """
    corpus = " ".join(texts)  # Create the corpus from all of the texts
    tokens = nltk.word_tokenize(corpus)
    frequencies = nltk.FreqDist(tokens)

    vocab_size = len(set(tokens))
    n_to_filter_out = int(proportion_words_to_filter_out * vocab_size)

    most_common_words = sorted(set([
        word for word, freq in frequencies.most_common(n_to_filter_out)
    ]))  # Most frequent words to filter out
    filtered_texts = remove_stopwords(
        texts,
        stopwords_list=most_common_words
    )

    return filtered_texts


def filter_out_too_frequent_words(
    data: pd.DataFrame,
    description_col: str,
    proportion_words_to_filter_out: float
) -> pd.DataFrame:
    """
    From the dataframe of the offers, removes the most frequent words
    from the descriptions independently for each language

    Args:
        data (pd.DataFrame): Dataframe of offers
        description_col (str): Name of the description column to treat
        proportion_words_to_filter_out (float)

    Returns:
        pd.DataFrame: Offers with a "trated" description column
    """
    languages_list = set(data["language"])
    well_described_data = data.copy()

    for language in tqdm(languages_list):

        # Filter the texts of only one language
        data_lang = data[data["language"] == language]
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
    threshold_short_text: int
) -> pd.DataFrame:
    """
    From given text columns, create new columns "beginning" and "end"
    corresponding to smaller extracts of the initial text columns

    Args:
        data (pd.DataFrame): The dataframe of offers
        cols_to_duplicate (list): The text columns to create extracts from
        threshold_short_text (int): Size of the desired extracts

    Returns:
        pd.DataFrame: Dataframe of offers with the new columns
    """
    data_with_new_cols = data.copy()

    for col in cols_to_duplicate:

        data_with_new_cols["beginning_"+col] = data_with_new_cols[
            col
        ].apply(lambda x: x[:threshold_short_text])
        data_with_new_cols["end_"+col] = data_with_new_cols[
            col
        ].apply(lambda x: x[-threshold_short_text:])

    return data_with_new_cols


def create_concatenated_column(
    data: pd.DataFrame,
    list_cols_to_concatenate: list,
    concatenated_col_name: str
) -> pd.DataFrame:
    """
    Concatenates a list of "interesting" columns into one new column

    Args:
        data (pd.DataFrame): The dataframe of offers
        list_cols_to_concatenate (list): The columns to concatenate
        concatenated_col_name (str): The name of the concatenated column

    Returns:
        pd.DataFrame: The dataframe of offers with the new column
    """
    data_with_new_cols = data.copy()
    data_with_new_cols[concatenated_col_name] = data[
        list_cols_to_concatenate[0]  # Needs at least one column in the list
    ]
    for col in tqdm(list_cols_to_concatenate[1:]):
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
    str_cols: dict,
    cols_to_concatenate: dict,
    languages_list: list,
    proportion_words_to_filter_out: float,
    threshold_short_text: int
) -> pd.DataFrame:
    """
    Performs an extensive preprocessing:
    - Removes special characters from text columns
    - Creates a new description column with no stopwords, lemmatized texts,
      and where most common words have been removed
    - Creates extracts of descriptions and concatenated columns

    Args:
        pre_preprocessed_data (pd.DataFrame): The slightly preprocessed offers
        str_cols (dict): The columns that can require cleaning
        cols_to_concatenate (dict): The "interesting" columns to concatenate
        languages_list (list): The list of languages observed in the offers
        proportion_words_to_filter_out (float)
        threshold_short_text (int): Size of the text columns extracts

    Returns:
        pd.DataFrame: The extensively preprocessed dataset
    """
    preprocessed_data = pre_preprocessed_data.copy()
    stopwords_list = create_stopwords_list(
        languages_list,
        without_accents=True
    )

    preprocessed_data[str_cols["normal"]] = preprocessed_data[
        str_cols["normal"]
    ].progress_apply(
        remove_special_characters
    )

    preprocessed_data["filtered_description"] = remove_stopwords(
        preprocessed_data["description"],
        stopwords_list=stopwords_list
    )

    preprocessed_data["filtered_description"] = lemmatize_texts(
        preprocessed_data["filtered_description"]
    )

    preprocessed_data = filter_out_too_frequent_words(
        preprocessed_data,
        description_col="filtered_description",
        proportion_words_to_filter_out=proportion_words_to_filter_out
    )

    preprocessed_data = create_extra_cols_from_text_cols(
        preprocessed_data,
        cols_to_duplicate=[
            "description",
            "filtered_description"
            ],
        threshold_short_text=threshold_short_text
    )

    preprocessed_data = create_concatenated_column(
        preprocessed_data,
        list_cols_to_concatenate=cols_to_concatenate['normal'],
        concatenated_col_name="concatenated_text"
    )

    preprocessed_data = create_concatenated_column(
        preprocessed_data,
        list_cols_to_concatenate=cols_to_concatenate['filtered'],
        concatenated_col_name="concatenated_filtered_text"
    )

    return preprocessed_data


def filter_international_companies(
    preprocessed_data: pd.DataFrame,
    cols_to_be_diversified: list
) -> pd.DataFrame:
    """
    Filters out "non-international" companies that have published offers
    in only one language and in only one country

    Args:
        preprocessed_data (pd.DataFrame): Dataset of offers
        cols_to_be_diversified (list): Columns with international aspect

    Returns:
        pd.DataFrame: International offers
    """
    # Count unique values in each of the columns of interest for each company
    unique_counts = preprocessed_data.groupby(
        ["company_name"]
    )[cols_to_be_diversified].nunique()

    # Filter out companies with only one unique value in each of them
    international_offers = preprocessed_data[
        preprocessed_data["company_name"].isin(
            unique_counts[(unique_counts > 1).any(axis=1)].index
        )
    ].reset_index(drop=True)

    print(
        f'Nb of offers by international companies: {len(international_offers)}'
    )
    return international_offers


def filter_out_poorly_described_offers(
    preprocessed_data: pd.DataFrame,
    cols_not_to_be_diversified: list
) -> pd.DataFrame:
    """
    Filters out offers for which the description is "generic", ie the offers
    for which the same description corresponds to several titles
    Also filters offers with no description at all

    Args:
        preprocessed_data (pd.DataFrame): Dataset of offers
        cols_not_to_be_diversified (list): Most of the time, ["title"]

    Returns:
        pd.DataFrame: _description_
    """
    preprocessed_data = preprocessed_data[
        preprocessed_data["filtered_description"] != ""
    ].reset_index(drop=True)

    data_with_one_col_not_to_be_diversified = create_concatenated_column(
        preprocessed_data,
        list_cols_to_concatenate=cols_not_to_be_diversified,
        concatenated_col_name="col_not_to_be_diversified"
    )

    # Count unique values for tuple of cols of interest for each description
    unique_counts = data_with_one_col_not_to_be_diversified.groupby(
        ["filtered_description"]
    )[["col_not_to_be_diversified"]].nunique()

    # Filter out companies with several unique values in any of them
    well_described_offers = preprocessed_data[
        preprocessed_data["filtered_description"].isin(
            unique_counts[(unique_counts == 1).any(axis=1)].index
        )
    ].reset_index(drop=True)

    print(
        f'Nb of not poorly described offers: {len(well_described_offers)}'
    )
    return well_described_offers


def ner_one_text(
    text: str
) -> set:
    """
    Computes NER of a text
    using distilbert-base-multilingual-cased-ner-hrl
    Keeps locations and companies of length > 3

    Args:
        text (str)

    Returns:
        set: Set of entities (location & companies) present in the text
    """
    raw_entities = ner_pipeline(text)

    filtered_entities = []
    end, index = 0, 0

    for entity in raw_entities:  # Post-processing required
        # We only keep locations and companies

        if entity['entity'] in ['B-LOC', 'B-ORG']:
            clean_entity = entity['word'].replace("##", "")
            end = entity['end']
            index = entity['index']
            filtered_entities.append(clean_entity)

        if (
            (entity['entity'] in ['I-LOC', 'I-ORG']) and
            (entity['index'] == index + 1) and
            (entity['start'] == end)
        ):
            beginning = filtered_entities.pop()
            clean_entity = beginning + entity['word'].replace("##", "")
            end = entity['end']
            index = entity['index']
            filtered_entities.append(clean_entity)

    return set([entity for entity in filtered_entities if len(entity) > 3])


def encode_ner(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates a new column "NER" corresponding to the set of entitites
    (loxations & companies) present in the concatenated offer

    Args:
        data (pd.DataFrame): Dataset of offers

    Returns:
        pd.DataFrame: Dataset of offers with the NER column
    """
    data["NER"] = data["concatenated_raw_text"].progress_apply(
        ner_one_text
    )

    data["NER"] = data["NER"].progress_apply(
        lambda x: set([unidecode(entity.lower()) for entity in x])
    )

    return data
