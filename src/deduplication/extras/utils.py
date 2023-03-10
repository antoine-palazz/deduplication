import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def compute_chunk_cosine_similarity(
    tokenized_texts,
    start: int,
    end: int
):
    try:
        end = min(end, tokenized_texts.shape[0])
    except AttributeError:
        end = min(end, len(tokenized_texts))
    cosine_similarity_matrix = cosine_similarity(
        X=tokenized_texts[start:end],
        Y=tokenized_texts[start:]
        )
    return cosine_similarity_matrix


def find_subtle_duplicates_from_tokens(
    data: pd.DataFrame,
    tokenized_texts,
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    chunk_size: int = 10000,
    threshold_semantic: int = 0.95,
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    duplicates = []
    try:
        n_ads = tokenized_texts.shape[0]
    except AttributeError:
        n_ads = len(tokenized_texts)

    for chunk_start in range(0, n_ads, chunk_size):
        similarity_matrix_chunk = compute_chunk_cosine_similarity(
            tokenized_texts,
            chunk_start,
            chunk_start+chunk_size)
        compteur_init = len(duplicates)

        for i in tqdm(range(chunk_size)):
            for j in range(i+1, n_ads):
                if similarity_matrix_chunk[i][j] > threshold_semantic:
                    if (data.loc[chunk_start+i, date_col] !=
                            data.loc[j, date_col]):
                        duplicates.append(
                            {'id1': data.loc[chunk_start+i, id_col],
                             'id2': data.loc[chunk_start+j, id_col],
                             'type': 'TEMPORAL'}
                        )
                    elif abs(  # Condition to change
                            len(data.loc[chunk_start+i, description_col]) -
                            len(data.loc[chunk_start+j, description_col])
                        ) / (1 + min(
                            len(data.loc[chunk_start+i, description_col]),
                            len(data.loc[chunk_start+j, description_col])
                            )) > threshold_partial:
                        duplicates.append(
                            {'id1': data.loc[chunk_start+i, id_col],
                             'id2': data.loc[chunk_start+j, id_col],
                             'type': 'PARTIAL'})
                    else:
                        duplicates.append(
                            {'id1': data.loc[chunk_start+i, id_col],
                             'id2': data.loc[chunk_start+j, id_col],
                             'type': 'SEMANTIC'})

        compteur_end = len(duplicates)
        print(
            f'{compteur_end-compteur_init} duplicates \
               found in chunck n??{int(chunk_start/chunk_size+1)}'
            )

    return(duplicates)
