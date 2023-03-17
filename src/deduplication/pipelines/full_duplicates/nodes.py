"""
This is a boilerplate pipeline 'full_duplicates'
generated using Kedro 0.18.6
"""

import pandas as pd
from tqdm import tqdm


def identify_full_duplicates(
    data: pd.DataFrame,
    type_to_return: str = 'FULL',
    list_cols_to_match: list = [['title', 'description']],
    backup_cols_to_match: list = ['company_name', 'location'],
    id_col: str = 'id'
) -> pd.DataFrame:

    if type_to_return in ['FULL', 'PARTIAL']:
        data = data[(data.str.len() == 0).sum() < 2]

    n_ads = len(data)
    full_duplicates = []

    for cols_to_match in tqdm(list_cols_to_match):
        data_for_duplicates = data.sort_values(
            by=cols_to_match+[id_col],
            ignore_index=True
        )
        cols_to_match_idxs = [
            data_for_duplicates.columns.get_loc(col)
            for col in cols_to_match
        ]
        backup_cols_to_match_idxs = [
            data_for_duplicates.columns.get_loc(col)
            for col in backup_cols_to_match
        ]
        data_for_dups_arr = data_for_duplicates.values

        for i in tqdm(range(n_ads-1)):
            j = i+1
            while (
                data_for_dups_arr[i, cols_to_match_idxs] ==
                data_for_dups_arr[j, cols_to_match_idxs]
            ).all() and (
                ((data_for_dups_arr[i, cols_to_match_idxs] != ''
                  ).all() and (
                    data_for_dups_arr[j, cols_to_match_idxs] != ''
                    ).all()
                 ) or (
                    (data_for_dups_arr[i, backup_cols_to_match_idxs] ==
                        data_for_dups_arr[j, backup_cols_to_match_idxs]
                     ).all()
                    )
            ):
                full_duplicates.append(
                    {
                        'id1': data_for_duplicates.loc[i, id_col],
                        'id2': data_for_duplicates.loc[j, id_col],
                        'type': type_to_return
                    })
                j += 1

    df_full_duplicates = pd.DataFrame(
        full_duplicates
    ).drop_duplicates(
    ).sort_values(by=['id1', 'id2'],
                  ignore_index=True)

    print(f'{len(df_full_duplicates)} {type_to_return} duplicates were found')
    return df_full_duplicates
