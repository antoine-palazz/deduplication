"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

import pandas as pd


def combine_all_duplicates(
    duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        [duplicates, subtle_duplicates],
        axis=0,
        ignore_index=True
    )

    all_duplicates.drop_duplicates(subset=['id1', 'id2'], inplace=True)
    all_duplicates.sort_values(by=['id1', 'id2'], inplace=True)

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
            ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def describe_duplicates(all_duplicates: pd.DataFrame) -> pd.DataFrame:
    duplicates_description = all_duplicates.groupby('type').count()
    return duplicates_description
