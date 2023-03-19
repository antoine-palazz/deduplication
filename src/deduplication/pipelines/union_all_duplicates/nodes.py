"""
This is a boilerplate pipeline 'union_all_duplicates'
generated using Kedro 0.18.6
"""

from kedro.config import ConfigLoader
from kedro.framework.project import settings
from kedro.io import DataCatalog
import pandas as pd
from tqdm import tqdm


def aggregate_duplicates_list(
    duplicates_list: list[pd.DataFrame]
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        duplicates_list
    ).drop_duplicates(
        subset=['id1', 'id2']
    ).sort_values(
        by=['id1', 'id2'],
        ignore_index=True
    ).reset_index(drop=True)

    return all_duplicates


def aggregate_easy_duplicates(
    preprocessed_data: pd.DataFrame,
    gross_full_duplicates: pd.DataFrame,
    gross_partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    cols_to_concatenate: list,
    id_col: str
) -> pd.DataFrame:

    easy_duplicates = aggregate_duplicates_list(
        [gross_full_duplicates,
         gross_partial_duplicates,
         gross_semantic_duplicates]
    )

    n_easy_duplicates = len(easy_duplicates)
    print(f'{n_easy_duplicates} "easy" duplicates isolated')
    if len(easy_duplicates[
        easy_duplicates['id1'] >= easy_duplicates['id2']
            ]) > 0:
        print('PROBLEM: id1 >= id2 in the "easy" duplicates table')

    # "Two identical partial 1 and 2 duplicates of the same record
    # (job posting) 3, are considered full duplicates"
    # Let's save the partial duplicates that can be changed into full ones!

    count_saved = 0
    partial_duplicates = easy_duplicates[easy_duplicates['type'] == 'PARTIAL']
    partial_id1s = set(partial_duplicates['id1'])
    dict_matchs_with_complete_offers = {}

    for id1 in tqdm(partial_id1s):
        # Fill a dictionary of the indexes that match with a complete offer
        dict_matchs_with_complete_offers[id1] = False

        if preprocessed_data[
            preprocessed_data[id_col] == id1
           ].iloc[0].apply(lambda x: x == "").sum() != 0:

            matchs_of_id1 = set(partial_duplicates[
                partial_duplicates['id1'] == id1
            ]['id2'])

            for id2 in matchs_of_id1:
                if preprocessed_data[
                    preprocessed_data[id_col] == id2
                   ].iloc[0].apply(lambda x: x == "").sum() == 0:

                    dict_matchs_with_complete_offers[id1] = True
                    break

    for idx_pair in tqdm(partial_duplicates.index):
        # Correct the identical pairs matching with at least one complete offer
        id1 = partial_duplicates.loc[idx_pair, 'id1']
        id2 = partial_duplicates.loc[idx_pair, 'id2']

        if dict_matchs_with_complete_offers[id1]:
            if (
                preprocessed_data[
                    preprocessed_data[id_col] == id1
                ].iloc[0][cols_to_concatenate] ==
                preprocessed_data[
                    preprocessed_data[id_col] == id2
                ].iloc[0][cols_to_concatenate]
               ).all():

                easy_duplicates.loc[idx_pair, 'type'] = 'FULL'
                count_saved += 1

    print(f'{count_saved} partial duplicates became full ones')

    return easy_duplicates


def aggregate_all_duplicates_one_model(
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates, subtle_duplicates]
    )

    n_all_duplicates = len(all_duplicates)
    print(f'{n_all_duplicates} duplicates isolated with this model')

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
            ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def aggregate_all_duplicates_several_models(
    easy_duplicates: pd.DataFrame,
    dummy_duplicates: pd.DataFrame,
    duplicates_str_list: list,
    project_path: str
) -> pd.DataFrame:

    conf_path = str(project_path + settings.CONF_SOURCE)
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    io = DataCatalog.from_config(conf_loader["catalog"])

    duplicates_list = [easy_duplicates] + [
        io.load(duplicates_str) for duplicates_str in duplicates_str_list
    ]

    all_duplicates = aggregate_duplicates_list(
        duplicates_list
    )

    n_all_duplicates = len(all_duplicates)
    print(f'{n_all_duplicates} duplicates combining all these models:')
    print(duplicates_str_list)

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
            ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def describe_duplicates(all_duplicates: pd.DataFrame) -> pd.DataFrame:
    duplicates_description = all_duplicates.groupby('type').count(
    ).reset_index()
    return duplicates_description
