"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

import pandas as pd


def combine_all_duplicates_one_model(
    full_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        [full_duplicates, subtle_duplicates],
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


def combine_all_duplicates_from_best_models(
    full_duplicates: pd.DataFrame,
    best_subtle_duplicates_temporal: pd.DataFrame,
    best_subtle_duplicates_partial: pd.DataFrame,
    best_subtle_duplicates_semantic: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = pd.concat(  # From most specific to least specific
        [
         full_duplicates,
         best_subtle_duplicates_temporal[
            best_subtle_duplicates_temporal['type'] == 'TEMPORAL'
            ],
         best_subtle_duplicates_partial[
            best_subtle_duplicates_partial['type'] == 'PARTIAL'
            ],
         best_subtle_duplicates_semantic[
            best_subtle_duplicates_semantic['type'] == 'SEMANTIC'
            ]
        ],
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
    duplicates_description = all_duplicates.groupby('type').count(
    ).reset_index()
    return duplicates_description
