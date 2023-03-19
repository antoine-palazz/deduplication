"""
This is a boilerplate pipeline 'union_all_duplicates'
generated using Kedro 0.18.6
"""

from kedro.config import ConfigLoader
from kedro.framework.project import settings
from kedro.io import DataCatalog
import pandas as pd


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
    gross_full_duplicates: pd.DataFrame,
    gross_partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame
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
