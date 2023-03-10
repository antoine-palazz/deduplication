"""
This is a boilerplate pipeline 'full_duplicates'
generated using Kedro 0.18.6
"""

import pandas as pd
from tqdm import tqdm


def identify_full_duplicates(
    data: pd.DataFrame,
    cols_to_match: list = ['title', 'description'],
    id_col: str = 'id'
) -> pd.DataFrame:

    full_duplicates = []
    data.sort_values(by=cols_to_match+[id_col], inplace=True)
    n_ads = len(data)

    for i in tqdm(range(n_ads-1)):
        j = i+1
        while (j < n_ads) and (
            (data.loc[i, cols_to_match] == data.loc[j, cols_to_match]).all()
        ):
            full_duplicates.append(
                {
                    'id1': data.loc[i, id_col],
                    'id2': data.loc[j, id_col],
                    'type': 'FULL'
                })
            j += 1

    print(len(full_duplicates))
    full_duplicates = pd.DataFrame(full_duplicates)
    return(full_duplicates)
