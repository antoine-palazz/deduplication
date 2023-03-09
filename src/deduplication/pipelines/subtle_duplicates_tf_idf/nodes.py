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

    end = min(end, tokenized_texts.shape[0])
    return(
        cosine_similarity(
            X=tokenized_texts[start:end],
            Y=tokenized_texts
        )
    )


def find_subtle_duplicates_from_tokens(
    data: pd.DataFrame,
    tokenized_texts,
    languages_list: list,
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    duplicates = []

    for chunk_start in range(0, len(data), chunk_size):
        similarity_matrix_chunk = compute_chunk_cosine_similarity(
            tokenized_texts,
            chunk_start,
            chunk_start+chunk_size)
        compteur_init = len(duplicates)

        for i in tqdm(range(chunk_size)):
            for j in range(chunk_start+i+1, len(data)):
                if similarity_matrix_chunk[i][j] > threshold_semantic:
                    if (data.loc[chunk_start+i, date_col] !=
                            data.loc[j, date_col]):
                        duplicates.append(
                            {'id1': data.loc[chunk_start+i, id_col],
                             'id2': data.loc[j, id_col], 'type': 'TEMPORAL'}
                        )
                    elif abs(
                            len(data.loc[chunk_start+i, description_col]) -
                            len(data.loc[j, description_col])
                        ) / (1 + min(
                            len(data.loc[chunk_start+i, description_col]),
                            len(data.loc[j, description_col])
                            )) > threshold_partial:
                        duplicates.append(
                            {'id1': data.loc[chunk_start+i, id_col],
                             'id2': data.loc[j, id_col], 'type': 'PARTIAL'})
                    else:
                        duplicates.append(
                            {'id1': data.loc[chunk_start+i, id_col],
                             'id2': data.loc[j, id_col], 'type': 'SEMANTIC'})

        compteur_end = len(duplicates)
        print(compteur_end-compteur_init)

    return(duplicates)


def identify_subtle_duplicates(
    data: pd.DataFrame,
    languages_list: list,
    concatenated_col_name: str = 'text',
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    max_df_tokenizer: float = 0.01,
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    stopwords_list = create_stopwords_list(languages_list)
    tokenized_texts = tokenize_tf_idf(
        data[concatenated_col_name],
        stopwords_list,
        max_df_tokenizer
    )
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

    return(pd.DataFrame(duplicates))
