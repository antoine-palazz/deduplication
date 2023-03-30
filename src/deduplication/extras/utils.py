import itertools
import sys
import uuid
from multiprocessing import Pool, cpu_count

import pandas as pd
from jellyfish import jaro_winkler_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def do_dates_differ_much(
    date_1,
    date_2,
    threshold_date: dict
) -> bool:

    dates_difference = abs((date_1 - date_2).days)

    if dates_difference > threshold_date["too_much"]:
        return "too_much"
    if dates_difference > threshold_date["far_dates"]:
        return "far_dates"
    return "close_dates"


def compare_text_lengths(
    text_1: str,
    text_2: str,
    lingual: str,
    thresholds_desc_len: dict
) -> str:

    absolute_lengths_diff = abs(len(text_1) - len(text_2))

    min_length = min(len(text_1), len(text_2))
    relative_lengths_diff = absolute_lengths_diff / (1 + min_length)

    if (
        (absolute_lengths_diff >
         thresholds_desc_len["absolute"][lingual]["NON"]) or
        (relative_lengths_diff >
            thresholds_desc_len["relative"][lingual]["NON"])
    ):

        return "too_long"

    if (
        (absolute_lengths_diff >
         thresholds_desc_len["absolute"][lingual]["PARTIAL"]) and
        (relative_lengths_diff >
            thresholds_desc_len["relative"][lingual]["PARTIAL"])
    ):

        return "different_lengths"

    return "same_length"


def is_full(
    row_1,
    row_2,
    current_type: str
) -> tuple[bool, str]:

    isfull = (
        (current_type == "FULL") and
        ((row_1.drop(["id", "retrieval_date"]) ==
          row_2.drop(["id", "retrieval_date"])).all())
    )
    if isfull:
        if row_1["retrieval_date"] == row_2["retrieval_date"]:
            return (True, "FULL")
        return (True, "TEMPORAL")

    return (False, "Unknown")


def is_non_duplicate(
    row_1,
    row_2,
    lingual: str,
    dates_differ: str,
    str_cols: dict,
    threshold_date: dict,
    thresholds_similarity: dict
) -> tuple[bool, str]:

    if row_1["country_id"] != row_2["country_id"]:
        return (True, "NON")  # To remove?

    if dates_differ == "too_much":
        return (True, "NON")  # To remove?

    # for col in str_cols["no_description"]:
    for col in str_cols["filtered"]:
        if row_1[col] != "" and row_2[col] != "":
            min_len_field = int(1.05*min(len(row_1[col]), len(row_2[col])))
            if (
                jaro_winkler_similarity(row_1[col], row_2[col])
                < thresholds_similarity[lingual][dates_differ][col]
            ) and (
                jaro_winkler_similarity(
                    row_1[col][:min_len_field],
                    row_2[col][:min_len_field]
                ) < thresholds_similarity[lingual][dates_differ][col]
            ):
                return (True, "NON")  # A field differs too much

    return (False, "Unknown")


def is_partial(
    row_1,
    row_2,
    lingual: str,
    dates_differ: str,
    str_cols: dict,
    thresholds_similarity: dict,
    thresholds_desc_len: dict
) -> tuple[bool, str]:

    if (
        (row_1[str_cols["no_description"]] ==
         row_2[str_cols["no_description"]]).all() and
        (row_1[str_cols["no_description"]] != "").all()
    ):
        return (False, "Unknown")  # Partial semantic (easy)

    lengths_differ = compare_text_lengths(
        row_1["filtered_description"],
        row_2["filtered_description"],
        lingual=lingual,
        thresholds_desc_len=thresholds_desc_len
    )

    if lengths_differ == "too_long":
        return (True, "NON")  # Description too long

    # START VERSION WITH NER HERE

    type_to_return = "Unknown"
    one_longer_than_two = (len(row_2["filtered_description"]) <
                           len(row_1["filtered_description"]))

    one_more_complete = 0
    two_more_complete = 0
    both_incomplete = 0

    for col in str_cols["filtered"]:
        if (row_1[col] == "") and (row_2[col] == ""):
            both_incomplete += 1
            if col == "filtered_description":
                both_incomplete += 1
        elif (row_1[col] == "") != (row_2[col] == ""):
            if (row_2[col] == "" and
                    row_1[col].split(" ", 1)[0] not in row_2["description"]):
                one_more_complete += 1
            if (row_1[col] == "" and
                    row_2[col].split(" ", 1)[0] not in row_1["description"]):
                two_more_complete += 1

    if one_more_complete + two_more_complete + both_incomplete == 0:
        return (False, "Unknown")  # Everything is already in the ad

    elif both_incomplete >= 3 or one_more_complete + two_more_complete >= 3:
        return (True, "NON")  # Too many empty fields

    elif one_more_complete * two_more_complete == 1:
        if lengths_differ == "different_lengths":
            type_to_return = "PARTIAL"  # Compensation by description length
        else:
            return (True, "NON")  # More than one field missing

    elif one_more_complete + two_more_complete >= 1:

        if one_more_complete == 1 and not (
            lengths_differ == "different_lengths" and
            not one_longer_than_two
        ):
            type_to_return = "PARTIAL"

        elif one_more_complete == 2 and (
            lengths_differ == "different_lengths" and
            not one_longer_than_two
        ):
            type_to_return = "PARTIAL"

        elif two_more_complete == 1 and not (
            lengths_differ == "different_lengths" and
            one_longer_than_two
        ):
            type_to_return = "PARTIAL"

        elif two_more_complete == 2 and (
            lengths_differ == "different_lengths" and
            one_longer_than_two
        ):
            type_to_return = "PARTIAL"

        elif one_more_complete + two_more_complete == 1:
            return (False, "Unknown")

        else:
            return (True, "NON")  # More than one field missing

    elif lengths_differ == "different_lengths":
        type_to_return = "PARTIAL"  # No info on missing fields but longer desc

    else:
        return (False, "Unknown")  # No info at all

    # END SEQUENCE WITH NER HERE

    if type_to_return == "PARTIAL":
        if row_1["retrieval_date"] != row_2["retrieval_date"]:
            return (True, "NON")  # PARTIAL + TEMPORAL = NON?
        return (True, "PARTIAL")

    return (False, "Unknown")


def differentiate_duplicates(
    row_1,
    row_2,
    lingual: str,
    dates_differ: str,
    current_type: str,
    str_cols: dict,
    threshold_date: dict,
    thresholds_similarity: dict,
    thresholds_desc_len: dict
) -> str:

    isfull, type_to_return = is_full(row_1, row_2, current_type=current_type)
    if isfull:
        return type_to_return

    isnon, type_to_return = is_non_duplicate(
        row_1,
        row_2,
        lingual=lingual,
        dates_differ=dates_differ,
        str_cols=str_cols,
        threshold_date=threshold_date,
        thresholds_similarity=thresholds_similarity
    )
    if isnon:
        return type_to_return

    ispartial, type_to_return = is_partial(
        row_1,
        row_2,
        lingual=lingual,
        dates_differ=dates_differ,
        str_cols=str_cols,
        thresholds_similarity=thresholds_similarity,
        thresholds_desc_len=thresholds_desc_len
    )
    if ispartial:
        return type_to_return

    if row_1["retrieval_date"] != row_2["retrieval_date"]:
        return "TEMPORAL"  # Dates are different

    return current_type  # No more tests, we go with the current assumption


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def reduce_dimension(matrix_texts: list, hyperparameters: dict) -> list:
    pca = TruncatedSVD(n_components=hyperparameters["dim_tokens"])
    reduced_embeddings = pca.fit_transform(matrix_texts)

    return reduced_embeddings


def compute_chunk_cosine_similarity(
    tokenized_texts,
    start: int,
    end: int
) -> list:
    try:
        end = min(end, tokenized_texts.shape[0])
    except AttributeError:
        end = min(end, len(tokenized_texts))
    cosine_similarity_matrix = cosine_similarity(
        X=tokenized_texts[start:end], Y=tokenized_texts[start:]
    )
    return cosine_similarity_matrix


def find_subtle_duplicates_from_tokens(
    data: pd.DataFrame,
    tokenized_texts: list,
    str_cols: dict,
    threshold_semantic: dict,
    threshold_date: dict,
    thresholds_similarity: dict,
    thresholds_desc_len: dict,
    hyperparameters: dict
) -> list:

    duplicates = []

    data_arr = data.values
    cols_idxs = {k: v for v, k in enumerate(data.columns)}

    n_ads = data_arr.shape[0]
    chunks = range(0, n_ads, hyperparameters["chunk_size"])

    for chunk_start in tqdm(chunks):

        similarity_matrix_chunk = compute_chunk_cosine_similarity(
            tokenized_texts,
            start=chunk_start,
            end=chunk_start+hyperparameters["chunk_size"]
        )

        def find_dups_in_chunk(i):
            if chunk_start + i >= n_ads:
                return []
            duplicates_chunk_i = []
            row_i = data_arr[chunk_start + i]

            for j in range(i + 1, n_ads - chunk_start):
                row_j = data_arr[chunk_start + j]

                ling = (
                    "monolingual" if (row_i[cols_idxs["language"]] ==
                                      row_j[cols_idxs["language"]])
                    else "multilingual"
                )
                dates_diff = do_dates_differ_much(
                    row_i[cols_idxs["retrieval_date"]],
                    row_j[cols_idxs["retrieval_date"]],
                    threshold_date=threshold_date
                    )

                if similarity_matrix_chunk[i][j] > threshold_semantic[ling][dates_diff]:
                    duplicates_type = differentiate_duplicates(
                        data.loc[chunk_start + i],
                        data.loc[chunk_start + j],
                        lingual=ling,
                        dates_differ=dates_diff,
                        current_type="SEMANTIC",
                        str_cols=str_cols,
                        threshold_date=threshold_date,
                        thresholds_similarity=thresholds_similarity,
                        thresholds_desc_len=thresholds_desc_len
                    )

                    if duplicates_type != "NON":
                        duplicates_chunk_i.append(
                            {
                                "id1": row_i[cols_idxs["id"]],
                                "id2": row_j[cols_idxs["id"]],
                                "type": duplicates_type,
                            }
                        )

            return duplicates_chunk_i

        find_dups_in_chunk_for_pickle = globalize(find_dups_in_chunk)
        with Pool(int(cpu_count()/2)) as pool:
            list_duplicates_chunk = pool.map(find_dups_in_chunk_for_pickle,
                                             range(hyperparameters["chunk_size"])
                                             )
        duplicates_chunk = list(
            itertools.chain.from_iterable(list_duplicates_chunk)
        )
        print(
            f"{len(duplicates_chunk)} duplicates \
               found in chunck n°{int(chunk_start/hyperparameters['chunk_size']+1)}"
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
