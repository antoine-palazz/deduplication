"""
This is a boilerplate pipeline 'easy_duplicates'
generated using Kedro 0.18.6
"""

import pandas as pd
from tqdm import tqdm


def identify_exact_duplicates(
    data: pd.DataFrame,
    list_cols_to_match: list,
    type_to_return: str,
    id_col: str = 'id'
) -> pd.DataFrame:

    n_ads = len(data)
    exact_duplicates = []

    for cols_to_match in tqdm(list_cols_to_match):

        data_for_duplicates = data.sort_values(
            by=cols_to_match+[id_col],
            ignore_index=True
        )

        cols_to_match_idxs = [
            data_for_duplicates.columns.get_loc(col)
            for col in cols_to_match
        ]
        data_for_dups_arr = data_for_duplicates.values

        for i in tqdm(range(n_ads-1)):
            j = i+1

            while (
                data_for_dups_arr[i, cols_to_match_idxs] ==
                data_for_dups_arr[j, cols_to_match_idxs]
            ).all():

                exact_duplicates.append(
                    {
                        'id1': data_for_duplicates.loc[i, id_col],
                        'id2': data_for_duplicates.loc[j, id_col],
                        'type': type_to_return
                    })
                j += 1

    df_exact_duplicates = pd.DataFrame(
        exact_duplicates
    ).drop_duplicates(
    ).sort_values(by=['id1', 'id2'],
                  ignore_index=True)

    print(
        f'{len(df_exact_duplicates)} gross {type_to_return} duplicates found'
    )
    return df_exact_duplicates
