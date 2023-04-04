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
) -> str:
    """
    Describe to what extent two dates differ

    Args:
        date_1 (datetime)
        date_2 (datetime)
        threshold_date (dict): The thresholds for dates

    Returns:
        str: A string indicating the closeness of the dates
    """
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
    """
    Describe to what extent two text lengths differ

    Arguments:
    text_1 (str)
    text_2 (str)
    lingual (str): "monolingual" / "multilingual"
    thresholds_desc_len (dict): The thresholds for text lengths

    Returns:
        str: A string indicating the closeness of the text lengths
    """
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


def compute_difference_sets(
    set_1: set,
    set_2: set
) -> set:
    """
    Computes the elements in set_1 that are not similar to any element in set_2

    Args:
        set_1 (set)
        set_2 (set)

    Returns:
        set: Pseudo difference between set_1 and set_2
    """
    new_set_1 = set([
        el for el in set_1 if all(
            [jaro_winkler_similarity(el, el_to_compare) < 0.8
             for el_to_compare in set_2]
        )
    ])
    return new_set_1


def is_full(
    row_1,
    row_2,
    current_type: str,
    str_cols: dict
) -> tuple[bool, str]:
    """
    Checks if two given rows are full duplicates.

    Args:
        row_1
        row_2
        current_type (str): current assumption for the type of duplicate
        str_cols (dict): columns to match to be considered a full duplicate

    Returns:
        tuple[bool, str]: (True, duplicate_type) / (False, "Unknown")
    """
    isfull = (
        (current_type == "FULL") and
        ((row_1[str_cols["normal"]] == row_2[str_cols["normal"]]).all())
    )
    if isfull:  # Apart for the date, it is a full duplicate
        if row_1["retrieval_date"] == row_2["retrieval_date"]:
            return (True, "FULL")
        return (True, "TEMPORAL")

    return (False, "Unknown")  # Not a full duplicate


def is_non_duplicate(
    row_1,
    row_2,
    lingual: str,
    dates_differ: str,
    str_cols: dict,
    thresholds_similarity: dict
) -> tuple[bool, str]:
    """
    Checks if a pair of rows can be labelled as non duplicates

    Args:
        row_1
        row_2
        lingual (str): "monolingual" / "multilingual"
        dates_differ (str): "close_dates" / "far_dates" / "too_much"
        str_cols (dict): Columns to compare
        thresholds_similarity (dict): Required thresholds for Jaro-Winkler

    Returns:
        tuple[bool, str]: (True, "NON") / (False, "Unknown")
    """
    if row_1["country_id"] != row_2["country_id"]:  # Same country_id required
        return (True, "NON")  # Is it relevant?

    if dates_differ == "too_much":  # Retrieval dates differ too much
        return (True, "NON")  # Is it relevant?

    for col in str_cols["normal"]:
        if row_1[col] != "" and row_2[col] != "":
            min_len_field = int(1.05*min(len(row_1[col]), len(row_2[col])))
            if (  # All columns must be similar to a certain extent
                jaro_winkler_similarity(row_1[col], row_2[col])
                < thresholds_similarity[lingual][dates_differ][col]
            ) and (  # At least until the minimal length of the two strings
                jaro_winkler_similarity(
                    row_1[col][:min_len_field],
                    row_2[col][:min_len_field]
                ) < thresholds_similarity[lingual][dates_differ][col]
            ):
                return (True, "NON")  # A field differs too much

    return (False, "Unknown")  # Possibly still a duplicate


def is_partial(
    row_1,
    row_2,
    lingual: str,
    dates_differ: str,
    str_cols: dict,
    thresholds_desc_len: dict
) -> tuple[bool, str]:
    """
    Checks if two given rows are partial duplicates.
    Method comparing the description lengths

    Args:
        row_1 (_type_)
        row_2 (_type_)
        lingual (str): "monolingual" / "multilingual"
        dates_differ (str): "close_dates" / "far_dates" / "too_much"
        str_cols (dict): Columns to compare
        thresholds_desc_len (dict): Thresholds to compare description lengths

    Returns:
        tuple[bool, str]: (True, duplicate_type) / (False, "Unknown")
    """
    if (
        (row_1[str_cols["no_description"]] ==
         row_2[str_cols["no_description"]]).all() and
        (row_1[str_cols["no_description"]] != "").all()
    ):
        return (False, "Unknown")  # All cols but the description are equal
        # And they are not empty, so not a partial

    lengths_differ = compare_text_lengths(
        row_1["filtered_description"],
        row_2["filtered_description"],
        lingual=lingual,
        thresholds_desc_len=thresholds_desc_len
    )

    if lengths_differ == "too_long":
        return (True, "NON")  # Description lengths differ too much

    # 2 options going from here:
    # - Count missing fields and compare description lengths (computed here)
    # - Alternative using NER (next function)

    type_to_return = "Unknown"
    one_longer_than_two = (len(row_2["filtered_description"]) <
                           len(row_1["filtered_description"]))

    one_more_complete = 0  # Nb of information row_1 has that row_2 does not
    two_more_complete = 0  # Nb of information row_2 has that row_1 does not
    both_incomplete = 0  # Nb of information that no one has

    for col in str_cols["filtered"]:  # Loop over all the columns to compare
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

    # Variables computed, let's start the comparisons:

    if one_more_complete + two_more_complete + both_incomplete == 0:
        return (False, "Unknown")  # No missing field in any offer

    elif both_incomplete >= 3 or one_more_complete + two_more_complete >= 3:
        return (True, "NON")  # Too many missing fields in the offers

    elif one_more_complete * two_more_complete == 1:
        # Each offer has exactly one info that the other offer does not
        if lengths_differ == "different_lengths":
            # A description compensates for one of the missing info
            type_to_return = "PARTIAL"  # In that case, PARTIAL
        else:
            return (True, "NON")  # More than one field missing, non duplicate

    elif one_more_complete + two_more_complete >= 1:
        # An offer has at least one more info than the other

        if one_more_complete == 1 and not (
            lengths_differ == "different_lengths" and
            not one_longer_than_two
        ):  # Offer 1 has one more piece of info than Offer 2
            # And Offer 2 does not compensate with its description
            type_to_return = "PARTIAL"

        elif one_more_complete == 2 and (
            lengths_differ == "different_lengths" and
            not one_longer_than_two
        ):  # Offer 1 has two more pieces of info than Offer 2
            # And we believe Offer 2 compensates for one through description
            type_to_return = "PARTIAL"

        elif two_more_complete == 1 and not (
            lengths_differ == "different_lengths" and
            one_longer_than_two
        ):  # Offer 2 has one more piece of info than Offer 1
            # And Offer 1 does not compensate with its description
            type_to_return = "PARTIAL"

        elif two_more_complete == 2 and (
            lengths_differ == "different_lengths" and
            one_longer_than_two
        ):  # Offer 2 has two more pieces of info than Offer 1
            # And we believe Offer 1 compensates for one through description
            type_to_return = "PARTIAL"

        elif one_more_complete + two_more_complete == 1:
            # If one information differs and case not caught before
            return (False, "Unknown")  # Not a partial but possibly semantic

        else:
            return (True, "NON")  # Too many fields of difference

    elif lengths_differ == "different_lengths":
        # 1 or 2 missing fields in both ads, but one is longer than the other
        # So it can compensate for 1 missing field through its description
        type_to_return = "PARTIAL"

    else:
        return (False, "Unknown")  # Possibly semantic with 1 or 2 cols missing

    # Both options (lengths comparison and NER) regroup here

    if type_to_return == "PARTIAL":  # Check if similare dates
        if row_1["retrieval_date"] != row_2["retrieval_date"]:
            return (True, "NON")  # PARTIAL + TEMPORAL = NON?
        return (True, "PARTIAL")


def is_partial_with_ner(
    row_1,
    row_2,
    lingual: str,
    str_cols: dict,
    thresholds_desc_len: dict
) -> tuple[bool, str]:
    """
    Checks if two given rows are partial duplicates.
    Method using NER

    Args:
        row_1 (_type_)
        row_2 (_type_)
        lingual (str): "monolingual" / "multilingual"
        dates_differ (str): "close_dates" / "far_dates" / "too_much"
        str_cols (dict): Columns to compare
        thresholds_desc_len (dict): Thresholds to compare description lengths

    Returns:
        tuple[bool, str]: (True, duplicate_type) / (False, "Unknown")
    """
    if (
        (row_1[str_cols["no_description"]] ==
         row_2[str_cols["no_description"]]).all() and
        (row_1[str_cols["no_description"]] != "").all()
    ):
        return (False, "Unknown")  # All cols but the description are equal
        # And they are not empty, so not a partial

    lengths_differ = compare_text_lengths(
        row_1["filtered_description"],
        row_2["filtered_description"],
        lingual=lingual,
        thresholds_desc_len=thresholds_desc_len
    )

    if lengths_differ == "too_long":
        return (True, "NON")  # Description lengths differ too much

    # 2 options going from here:
    # - Count missing fields and compare description lengths (previous func)
    # - Alternative using NER (computed here)

    if (
        (row_1[str_cols["filtered"]] != "").all() and
        (row_2[str_cols["filtered"]] != "").all()
    ):
        return (False, "Unknown")  # No empty field -> not a partial

    # We compute the pseudo difference (using Jaro Winkler)
    # in the set of entities (location and companies) in each offer

    ner_row_1 = set(row_1["NER"])
    ner_row_2 = set(row_2["NER"])

    filtered_ner_1 = ner_row_1.difference(ner_row_2)
    filtered_ner_2 = ner_row_2.difference(ner_row_1)

    unique_ner_1 = compute_difference_sets(filtered_ner_1, filtered_ner_2)
    unique_ner_2 = compute_difference_sets(filtered_ner_2, filtered_ner_1)

    # Now let's start with the comparisons:

    if len(unique_ner_1) == 0 and len(unique_ner_2) == 0:
        return (False, "Unknown")  # Same entities present in each row

    if len(unique_ner_1) == len(unique_ner_2):
        return (False, "Unknown")  # As much information in each ad

    if lengths_differ == "same_size" and lingual == "multilingual":
        return (False, "Unknown")  # Hard to conclude on multilingual

    type_to_return = "PARTIAL"  # Otherwise, this is a PARTIAL

    # Both options (lengths comparison and NER) regroup here

    if type_to_return == "PARTIAL":  # Check if similare dates
        if row_1["retrieval_date"] != row_2["retrieval_date"]:
            return (True, "NON")  # PARTIAL + TEMPORAL = NON?
        return (True, "PARTIAL")


def differentiate_duplicates(
    row_1,
    row_2,
    lingual: str,
    dates_differ: str,
    current_type: str,
    str_cols: dict,
    threshold_date: dict,
    thresholds_similarity: dict,
    thresholds_desc_len: dict,
    ner: dict
) -> str:
    """
    Given a pair of potential duplicates, checks if they indeed are
    If they are, also returns the type of duplicates

    Args:
        row_1 (_type_)
        row_2 (_type_)
        lingual (str): "monolingual" / "multilingual"
        dates_differ (str): "close_dates" / "far_dates" / "too_much"
        current_type (str): current assumption for the type of duplicate
        str_cols (dict): Columns to compare
        threshold_date (dict): The thresholds for dates
        thresholds_similarity (dict): The thresholds for text lengths
        thresholds_desc_len (dict): Thresholds to compare description lengths
        ner (dict): booleans to know whether to use NER or not

    Returns:
        str: type of duplicates (or "NON")
    """

    isfull, type_to_return = is_full(
        row_1,
        row_2,
        current_type=current_type,
        str_cols=str_cols
    )
    if isfull:  # Do we have a full duplicate?
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
    if isnon:  # Do we have a duplicate at all?
        return type_to_return

    if ner["compute"] and ner["use"]:  # If we choose NER
        ispartial, type_to_return = is_partial_with_ner(
            row_1,
            row_2,
            lingual=lingual,
            str_cols=str_cols,
            thresholds_desc_len=thresholds_desc_len
        )
    else:  # If we choose to use descriptions lengths
        ispartial, type_to_return = is_partial(
            row_1,
            row_2,
            lingual=lingual,
            dates_differ=dates_differ,
            str_cols=str_cols,
            thresholds_desc_len=thresholds_desc_len
        )

    if ispartial:  # Do we have a partial duplicate?
        return type_to_return

    # By then, only semantic duplicates should remain.

    if row_1["retrieval_date"] != row_2["retrieval_date"]:
        return "TEMPORAL"  # Dates are different

    # Unless we find more tests to make, we go with the current assumption
    return current_type


def globalize(func):
    """
    "Globalizes" a nested function so that it can be multiprocessed

    Args:
        func: Function to globalize

    Returns:
        A globalized function
    """
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def reduce_dimension(matrix_texts: list, hyperparameters: dict) -> list:
    """
    Reduces the dimension of a list of vectors
    to make cosine similarity comparisons between them more relevant

    Args:
        matrix_texts (list): List of high-dimension tokens
        hyperparameters (dict): Includes the desired size for the final tokens

    Returns:
        list: List of lower-dimension tokens
    """
    pca = TruncatedSVD(n_components=hyperparameters["dim_tokens"])
    reduced_embeddings = pca.fit_transform(matrix_texts)

    return reduced_embeddings


def compute_chunk_cosine_similarity(
    tokenized_texts,
    start: int,
    end: int
) -> list:
    """
    Computes a chunk of cosine similarity matrix

    Args:
        tokenized_texts (list): List of tokens
        start (int): Beginning of chunk
        end (int): End of chunk

    Returns:
        list: Chunk of cosine similarity matrix
    """
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
    hyperparameters: dict,
    ner: dict
) -> list:
    """
    From a list of tokens,
    computes the list of duplicates based on cosine similarity comparisons

    Args:
        data (pd.DataFrame): DataFrame of the offers
        tokenized_texts (list): Embeddings of the concatenated offers
        str_cols (dict): Columns to compare
        threshold_semantic (dict): Thresholds for cosine similarities
        threshold_date (dict): The thresholds for dates
        thresholds_similarity (dict): The thresholds for text lengths
        thresholds_desc_len (dict): Thresholds to compare description lengths
        hyperparameters (dict): Diverse hyperparameters, including chunk size
        ner (dict): booleans to know whether to use NER or not

    Returns:
        list: Duplicates found with the embeddings
    """

    duplicates = []

    data_arr = data.values  # Convert data to np array for faster computations
    cols_idxs = {k: v for v, k in enumerate(data.columns)}

    n_ads = data_arr.shape[0]
    chunks = range(0, n_ads, hyperparameters["chunk_size"])

    for chunk_start in tqdm(chunks):
        # The whole cosine similarity matrix is too big to compute
        # So we do it by chunks

        similarity_matrix_chunk = compute_chunk_cosine_similarity(
            tokenized_texts,
            start=chunk_start,
            end=chunk_start+hyperparameters["chunk_size"]
        )

        def find_dups_in_chunk(i):  # Nested func to find duplicates in chunk
            # Each call looks for the duplicates with a given offer in chunk
            if chunk_start + i >= n_ads:
                return []
            duplicates_chunk_i = []
            row_i = data_arr[chunk_start + i]

            for j in range(i + 1, n_ads - chunk_start):
                # For all j > i, we test if (i,j) is a duplicates pair
                row_j = data_arr[chunk_start + j]

                ling = (
                    "monolingual" if (row_i[cols_idxs["language"]] ==
                                      row_j[cols_idxs["language"]])
                    else "multilingual"
                )  # Are the offers in the same languages?
                dates_diff = do_dates_differ_much(
                    row_i[cols_idxs["retrieval_date"]],
                    row_j[cols_idxs["retrieval_date"]],
                    threshold_date=threshold_date
                    )  # Are the retrieval dates close or not?

                if (
                    similarity_matrix_chunk[i][j] >
                    threshold_semantic[ling][dates_diff]
                ):  # If the cosine similarity passes the thereshold
                    duplicates_type = differentiate_duplicates(
                        data.loc[chunk_start + i],
                        data.loc[chunk_start + j],
                        lingual=ling,
                        dates_differ=dates_diff,
                        current_type="SEMANTIC",
                        str_cols=str_cols,
                        threshold_date=threshold_date,
                        thresholds_similarity=thresholds_similarity,
                        thresholds_desc_len=thresholds_desc_len,
                        ner=ner
                    )  # Let's go deeper into the pair

                    if duplicates_type != "NON":
                        # If the pair was indeed found to be a duplicate
                        duplicates_chunk_i.append(
                            {
                                "id1": row_i[cols_idxs["id"]],
                                "id2": row_j[cols_idxs["id"]],
                                "type": duplicates_type,
                            }
                        )

            return duplicates_chunk_i  # All duplicates j of the offer i (j>i)

        find_dups_in_chunk_for_pickle = globalize(find_dups_in_chunk)
        with Pool(int(cpu_count()/2)) as pool:  # Multiprocessing
            list_duplicates_chunk = pool.map(
                find_dups_in_chunk_for_pickle,
                range(hyperparameters["chunk_size"])
            )
        duplicates_chunk = list(
            itertools.chain.from_iterable(list_duplicates_chunk)
        )  # All duplicates in the chunk
        print(
            f"{len(duplicates_chunk)} duplicates \
                in chunck n°{int(chunk_start/hyperparameters['chunk_size']+1)}"
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
