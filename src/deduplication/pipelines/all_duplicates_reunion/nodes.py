"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    differentiate_semantic_duplicates
)
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from kedro.io import DataCatalog
import pandas as pd


def differentiate_gross_semantic_duplicates(
    data: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    full_duplicates: pd.DataFrame,
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    threshold_partial: int = 0.1
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        [full_duplicates, gross_semantic_duplicates],
    ).drop_duplicates()
    true_gross_semantic_duplicates = all_duplicates[
        all_duplicates["type"] != "FULL"
    ].reset_index(drop=True)

    print(f'{len(true_gross_semantic_duplicates)} "easy" duplicates to affect')

    n_gross_duplicates = len(true_gross_semantic_duplicates)
    for pair_id in range(n_gross_duplicates):

        row_1 = dict(data[
                    (data[id_col] ==
                     (true_gross_semantic_duplicates.loc[pair_id]["id1"]))
                ])
        row_2 = dict(data[
                    (data[id_col] ==
                     (true_gross_semantic_duplicates.loc[pair_id]["id2"]))
                ])

        duplicates_type = differentiate_semantic_duplicates(
                        row_1,
                        row_2,
                        description_col,
                        date_col,
                        threshold_partial
                    )

        true_gross_semantic_duplicates.loc[pair_id]["type"] = duplicates_type

    return true_gross_semantic_duplicates


def combine_all_duplicates_one_model(
    full_duplicates: pd.DataFrame,
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        [full_duplicates, easy_duplicates, subtle_duplicates],
        axis=0
    )

    all_duplicates.drop_duplicates(subset=['id1', 'id2'], inplace=True)
    all_duplicates.sort_values(by=['id1', 'id2'],
                               inplace=True,
                               ignore_index=True)

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
            ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def combine_all_duplicates_from_best_models(
    full_duplicates: pd.DataFrame,
    easy_duplicates: pd.DataFrame,
    best_model_temporal: str,
    best_model_partial: str,
    best_model_semantic: str,
    str_subtle_duplicates: str = 'subtle_duplicates',
    project_path: str = '/home/onyxia/work/deduplication/'
) -> pd.DataFrame:

    conf_path = str(project_path + settings.CONF_SOURCE)
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    io = DataCatalog.from_config(conf_loader["catalog"])

    best_subtle_duplicates_temporal = io.load(
        str_subtle_duplicates + best_model_temporal
        )
    best_subtle_duplicates_partial = io.load(
        str_subtle_duplicates + best_model_partial
        )
    best_subtle_duplicates_semantic = io.load(
        str_subtle_duplicates + best_model_semantic
        )

    all_duplicates = pd.concat(  # From most specific to least specific
        [
         full_duplicates,
         easy_duplicates,
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
        axis=0
    )

    all_duplicates.drop_duplicates(subset=['id1', 'id2'], inplace=True)
    all_duplicates.sort_values(by=['id1', 'id2'],
                               inplace=True,
                               ignore_index=True)

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
            ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def describe_duplicates(all_duplicates: pd.DataFrame) -> pd.DataFrame:
    duplicates_description = all_duplicates.groupby('type').count(
    ).reset_index()
    return duplicates_description
