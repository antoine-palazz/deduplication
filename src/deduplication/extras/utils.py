import itertools
import sys
import uuid
from multiprocessing import Pool, cpu_count

import pandas as pd
from jellyfish import jaro_winkler_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def reduce_dimension(matrix_texts: list, dim_tokens: int) -> list:
    pca = TruncatedSVD(n_components=dim_tokens)
    reduced_embeddings = pca.fit_transform(matrix_texts)

    return reduced_embeddings


def compute_chunk_cosine_similarity(tokenized_texts, start: int, end: int):
    try:
        end = min(end, tokenized_texts.shape[0])
    except AttributeError:
        end = min(end, len(tokenized_texts))
    cosine_similarity_matrix = cosine_similarity(
        X=tokenized_texts[start:end], Y=tokenized_texts[start:]
    )
    return cosine_similarity_matrix


def differentiate_duplicates(
    row_1,
    row_2,
    current_type: str,
    str_cols: list,
    cols_to_be_similar: list,
    title_col: str,
    description_col: str,
    date_col: str,
    language_col: str,
    threshold_similarity: dict,
    threshold_partial: float,
) -> str:

    for col in cols_to_be_similar:
        if row_1[col] != "" and row_2[col] != "":
            if (
                jaro_winkler_similarity(row_1[col], row_2[col])
                < threshold_similarity[col]
            ):
                return "NON"  # Desired columns are too different

    if row_1[language_col] == row_2[language_col]:
        if (
            jaro_winkler_similarity(row_1[title_col], row_2[title_col])
            < threshold_similarity['title_monolingual']
           ):
            return "NON"  # If same language, higher threshold for title

    if row_1[date_col] != row_2[date_col]:
        return "TEMPORAL"  # Dates are different

    count_partial = 0
    for col in str_cols:
        if (
            row_1[col] == "" or row_2[col] == "") and (
            row_1[col] != row_2[col]
        ):
            count_partial += 1
    if count_partial == 1:
        return "PARTIAL"  # One field is missing in only one of the ads

    # if (
    #     abs(len(row_1[description_col]) - len(row_2[description_col]))
    #     / (1 + min(len(row_1[description_col]), len(row_2[description_col])))
    #     > threshold_partial
    # ):
    #     return "PARTIAL"  # Description lengths are too different

    if current_type != 'FULL':
        return 'SEMANTIC'  # Can only be semantic by that point

    return 'FULL'


def find_subtle_duplicates_from_tokens(
    data: pd.DataFrame,
    tokenized_texts: list,
    str_cols: list,
    cols_to_be_similar: list,
    title_col: str,
    description_col: str,
    date_col: str,
    id_col: str,
    language_col: str,
    threshold_similarity: dict,
    threshold_semantic: float,
    threshold_partial: float,
    chunk_size: int
) -> list:

    duplicates = []

    n_ads = len(data)
    chunks = range(0, n_ads, chunk_size)
    n_chunks = len(chunks)

    for chunk_start in tqdm(chunks):

        similarity_matrix_chunk = compute_chunk_cosine_similarity(
            tokenized_texts,
            start=chunk_start,
            end=chunk_start+chunk_size
        )

        def find_dups_in_chunk(i):
            duplicates_chunk_i = []
            for j in range(i + 1, n_ads - chunk_start):
                if similarity_matrix_chunk[i][j] > threshold_semantic:
                    duplicates_type = differentiate_duplicates(
                        data.loc[chunk_start + i],
                        data.loc[chunk_start + j],
                        current_type="SEMANTIC",
                        str_cols=str_cols,
                        cols_to_be_similar=cols_to_be_similar,
                        title_col=title_col,
                        description_col=description_col,
                        date_col=date_col,
                        language_col=language_col,
                        threshold_similarity=threshold_similarity,
                        threshold_partial=threshold_partial,
                    )

                    if duplicates_type != "NON":
                        duplicates_chunk_i.append(
                            {
                                "id1": data.loc[chunk_start + i, id_col],
                                "id2": data.loc[chunk_start + j, id_col],
                                "type": duplicates_type,
                            }
                        )

            return duplicates_chunk_i

        find_dups_in_chunk_for_pickle = globalize(find_dups_in_chunk)
        with Pool(int(cpu_count()/3)) as pool:
            list_duplicates_chunk = pool.map(find_dups_in_chunk_for_pickle,
                                             range(chunk_size)
                                             )
        duplicates_chunk = list(
            itertools.chain.from_iterable(list_duplicates_chunk)
        )
        print(
            f"{len(duplicates_chunk)} duplicates \
               found in chunck n°{int(chunk_start/chunk_size+1)} / {n_chunks}"
        )
        duplicates.extend(duplicates_chunk)

    df_duplicates = (
        pd.DataFrame(duplicates)
        .drop_duplicates()
        .sort_values(by=["id1", "id2"], ignore_index=True)
    )
    print(
        f"{len(df_duplicates)} gross subtle duplicates found"
    )
    return df_duplicates
