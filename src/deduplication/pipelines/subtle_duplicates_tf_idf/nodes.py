"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

nltk.download('stopwords')
tqdm.pandas()
warnings.filterwarnings('ignore')


def create_stopwords_list(languages_list: list) -> list:
    stopwords_list = stopwords.words(languages_list)
    return stopwords_list


def tokenize_tf_idf(
    texts: pd.Series,
    stopwords_list: list,
    max_df_tokenizer: float = 0.01
):

    vectorizer = TfidfVectorizer(
        stop_words=stopwords_list,
        max_df=max_df_tokenizer
    )

    tokenized_texts = vectorizer.fit_transform(texts)
    return tokenized_texts


def compute_chunk_cosine_similarity(
    tokenized_texts,
    start: int,
    end: int
):

    end = max(end, len(tokenized_texts))
    return(
        cosine_similarity(
            X=tokenized_texts[start:end],
            Y=tokenized_texts
        )
    )


def identify_subtle_duplicates(
    data: pd.DataFrame,
    languages_list: list,
    id_col: str = 'id'
) -> pd.DataFrame:

    duplicates = []



    data.sort_values(by=cols_to_match + [id_col], inplace=True)
    n_ads = len(data)

    for i in tqdm(range(n_ads-1)):
        j = i+1
        while (j < n_ads) and (
            (data.loc[i, cols_to_match] == data.loc[j, cols_to_match]).all()
        ):
            full_duplicates.append(
                {
                    'id1': data.loc[i, id_col],
                    'id2': data.loc[j, id_col],
                    'type': 'FULL'
                })
            j += 1

    return(pd.DataFrame(full_duplicates))
