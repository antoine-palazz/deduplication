"""
This is a boilerplate pipeline 'union_all_duplicates'
generated using Kedro 0.18.6
"""

import pandas as pd


def aggregate_duplicates_list(
    duplicates_list: list[pd.DataFrame]
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        duplicates_list
    ).drop_duplicates(
        subset=['id1', 'id2']
    ).sort_values(
        by=['id1', 'id2']
    ).reset_index(drop=True)

    return all_duplicates


def aggregate_easy_duplicates(
    preprocessed_data: pd.DataFrame,
    gross_full_duplicates: pd.DataFrame,
    gross_partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    gross_semantic_multilingual_duplicates: pd.DataFrame,
    cols_to_concatenate: dict,
    id_col: str
) -> pd.DataFrame:

    easy_duplicates = aggregate_duplicates_list(
        [gross_full_duplicates,
         gross_partial_duplicates,
         gross_semantic_duplicates,
         gross_semantic_multilingual_duplicates]
    )

    n_easy_duplicates = len(easy_duplicates)
    print(f'{n_easy_duplicates} "easy" duplicates isolated:')
    print(describe_duplicates(easy_duplicates))

    if len(easy_duplicates[
        easy_duplicates['id1'] >= easy_duplicates['id2']
            ]) > 0:
        print('PROBLEM: id1 >= id2 in the "easy" duplicates table')

    return easy_duplicates


def print_true_subtle_duplicates(
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_subtle_pairs = subtle_duplicates[['id1', 'id2']].merge(
        easy_duplicates[['id1', 'id2']],
        on=['id1', 'id2'],
        how='left',
        indicator=True
    )

    true_subtles_duplicates = subtle_duplicates[
        ~all_subtle_pairs['_merge'].isin(['both'])
    ]

    print(
        f'{len(true_subtles_duplicates)} duplicates isolated with this model:'
    )
    print(describe_duplicates(true_subtles_duplicates))


def aggregate_all_duplicates_one_model(
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates, subtle_duplicates]
    )

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
    ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    print_true_subtle_duplicates(easy_duplicates, subtle_duplicates)
    return all_duplicates


def aggregate_all_duplicates_several_models(
    **kwargs
) -> pd.DataFrame:

    easy_duplicates = kwargs['easy']

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates] +
        [duplicates for name, duplicates in kwargs.items()]
    )

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
    ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def describe_duplicates(all_duplicates: pd.DataFrame) -> pd.DataFrame:
    duplicates_description = all_duplicates.groupby('type').count(
    ).reset_index()
    return duplicates_description
