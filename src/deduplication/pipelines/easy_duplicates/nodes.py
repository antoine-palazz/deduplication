"""
This is a boilerplate pipeline 'easy_duplicates'
generated using Kedro 0.18.6
"""

import pandas as pd
from tqdm import tqdm

from deduplication.extras.utils import differentiate_duplicates, do_dates_differ_much


def identify_exact_duplicates(
    data: pd.DataFrame,
    list_cols_to_match: dict,
    list_cols_to_mismatch: dict,
    default_type: str,
    str_cols: dict,
    threshold_date: int,
    thresholds_similarity: dict,
    thresholds_desc_len: dict
) -> pd.DataFrame:

    n_ads = len(data)
    exact_duplicates = []

    for cols_to_match in tqdm(list_cols_to_match[default_type]):
        for cols_to_mismatch in list_cols_to_mismatch[default_type]:

            data_for_duplicates = data.sort_values(
                by=cols_to_match+["id"],
                ignore_index=True
            )

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
                j = i+1

                while (
                    data_for_dups_arr[i, cols_to_match_idxs] ==
                    data_for_dups_arr[j, cols_to_match_idxs]
                ).all():

                    if (
                        data_for_dups_arr[i, cols_to_mismatch_idxs] !=
                        data_for_dups_arr[j, cols_to_mismatch_idxs]
                    ).all():

                        lingual = (
                            "monolingual" if (data_for_duplicates.loc[i]["language"] ==
                                              data_for_duplicates.loc[j]["language"])
                            else "multilingual"
                        )
                        dates_differ = (
                            "far_dates" if do_dates_differ_much(
                                data_for_duplicates.loc[i]["retrieval_date"],
                                data_for_duplicates.loc[j]["retrieval_date"],
                                threshold_date=threshold_date
                            ) else "close_dates"
                        )

                        duplicates_type = differentiate_duplicates(
                            data_for_duplicates.loc[i],
                            data_for_duplicates.loc[j],
                            lingual=lingual,
                            dates_differ=dates_differ,
                            current_type=default_type.split("_", 1)[0],
                            str_cols=str_cols,
                            threshold_date=threshold_date,
                            thresholds_similarity=thresholds_similarity,
                            thresholds_desc_len=thresholds_desc_len
                        )

                        if duplicates_type != "NON":
                            exact_duplicates.append(
                                {
                                    'id1': data_for_duplicates.loc[i, "id"],
                                    'id2': data_for_duplicates.loc[j, "id"],
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
