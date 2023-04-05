"""
This is a boilerplate pipeline 'easy_duplicates'
generated using Kedro 0.18.6
"""

import pandas as pd
from tqdm import tqdm

from deduplication.extras.utils import (
    differentiate_duplicates,
    do_dates_differ_much
)


def identify_exact_duplicates(
    data: pd.DataFrame,
    list_cols_to_match: dict,
    list_cols_to_mismatch: dict,
    default_type: str,
    str_cols: dict,
    thresholds_dates: dict,
    thresholds_similarity: dict,
    thresholds_desc_len: dict,
    ner: dict
) -> pd.DataFrame:
    """
    "Easy" approach to identify duplicated offers, by testing
    if certain sets of columns match or mismatch, which could
    provide some obvious duplicates

    Args:
        data (pd.DataFrame): Dataset of offers
        list_cols_to_match (dict): Columns required to match
        list_cols_to_mismatch (dict): Columns required to mismatch
        default_type (str): Current assumption about the duplicates type
        str_cols (dict): Columns to compare via Jaro Winkler
        thresholds_dates (dict): Thresholds for the dates
        thresholds_similarity (dict): Thresholds for the cosine similarities
        thresholds_desc_len (dict): Thresholds for the description lengths
        ner (dict): Booleans to decide if we use NER or not

    Returns:
        pd.DataFrame: Dataframe of duplicates and their types
    """
    n_ads = len(data)
    exact_duplicates = []

    for cols_to_match in tqdm(list_cols_to_match[default_type]):
        for cols_to_mismatch in list_cols_to_mismatch[default_type]:

            # Sort the table alphabetically by the cols to match (O(n*log(n)))
            # To reduce the following search of duplicates from O(n^2) to O(n)
            data_for_duplicates = data.sort_values(
                by=cols_to_match+["id"],
                ignore_index=True
            )

            # Converts dataframes to numpy array to run faster
            cols_to_match_idxs = [
                data_for_duplicates.columns.get_loc(col)
                for col in cols_to_match
            ]
            cols_to_mismatch_idxs = [
                data_for_duplicates.columns.get_loc(col)
                for col in cols_to_mismatch
            ]
            data_for_dups_arr = data_for_duplicates.values

            for i in tqdm(range(n_ads-1)):
                row_i = data_for_duplicates.loc[i]
                j = i+1

                while (
                    data_for_dups_arr[i, cols_to_match_idxs] ==
                    data_for_dups_arr[j, cols_to_match_idxs]
                ).all():

                    if (
                        data_for_dups_arr[i, cols_to_mismatch_idxs] !=
                        data_for_dups_arr[j, cols_to_mismatch_idxs]
                    ).all():
                        # By then we have a possible match between two rows
                        # Need to go deeper into the investigation

                        row_j = data_for_duplicates.loc[j]
                        lingual = (
                            "monolingual" if (row_i["language"] ==
                                              row_j["language"])
                            else "multilingual"
                        )
                        dates_differ = do_dates_differ_much(
                            row_i["retrieval_date"],
                            row_j["retrieval_date"],
                            thresholds_dates=thresholds_dates
                            )

                        duplicates_type = differentiate_duplicates(
                            row_i,
                            row_j,
                            lingual=lingual,
                            dates_differ=dates_differ,
                            current_type=default_type.split("_", 1)[0],
                            str_cols=str_cols,
                            thresholds_dates=thresholds_dates,
                            thresholds_similarity=thresholds_similarity,
                            thresholds_desc_len=thresholds_desc_len,
                            ner=ner
                        )

                        if duplicates_type != "NON":
                            exact_duplicates.append(
                                {
                                    'id1': row_i["id"],
                                    'id2': row_j["id"],
                                    'type': duplicates_type
                                })

                    j += 1

    df_exact_duplicates = pd.DataFrame(
        exact_duplicates
    ).drop_duplicates(
    ).sort_values(
        by=['id1', 'id2'],
        ignore_index=True
    )
    print(
        f'{len(df_exact_duplicates)} gross {default_type} duplicates found'
    )
    return df_exact_duplicates
