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
    id_col: str = 'id'
) -> pd.DataFrame:

    n_ads = len(data)
    full_duplicates = []

    for cols_to_match in tqdm(list_cols_to_match):
        data.sort_values(by=cols_to_match+[id_col],
                         inplace=True,
                         ignore_index=True)
        print(data.head())
        for i in tqdm(range(n_ads-1)):
            j = i+1
            while (j < n_ads) and (
                (
                    data.loc[i, cols_to_match] == data.loc[j, cols_to_match]
                ).all()
            ):
                full_duplicates.append(
                    {
                        'id1': data.loc[i, id_col],
                        'id2': data.loc[j, id_col],
                        'type': type_to_return
                    })
                j += 1

    full_duplicates = pd.DataFrame(full_duplicates).drop_duplicates()

    print(f'{len(full_duplicates)} {type_to_return} duplicates were found')
    return(full_duplicates)
