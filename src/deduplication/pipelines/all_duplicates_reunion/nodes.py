"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

from deduplication.extras.utils import (
    differentiate_easy_duplicates
)
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from kedro.io import DataCatalog
import pandas as pd
from tqdm import tqdm


def differentiate_df_easy_duplicates(
    data: pd.DataFrame,
    full_duplicates: pd.DataFrame,
    partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    description_col: str = 'description',
    date_col: str = 'retrieval_date',
    id_col: str = 'id',
    threshold_partial: float = 0.1
) -> pd.DataFrame:

    easy_duplicates = pd.concat(
        [full_duplicates, partial_duplicates, gross_semantic_duplicates],
    ).drop_duplicates(subset=['id1', 'id2']
                      ).reset_index(drop=True)

    n_easy_duplicates = len(easy_duplicates)
    print(f'{n_easy_duplicates} "easy" duplicates to affect')

    for pair_id in tqdm(range(n_easy_duplicates)):

        id1 = easy_duplicates.loc[pair_id]["id1"]
        row1 = data[
                    data[id_col] == id1
                ].reset_index(drop=True).loc[0]

        id2 = easy_duplicates.loc[pair_id]["id2"]
        row2 = data[
                    data[id_col] == id2
                ].reset_index(drop=True).loc[0]

        duplicates_type = differentiate_easy_duplicates(
                        row1,
                        row2,
                        easy_duplicates.loc[pair_id, "type"],
                        description_col,
                        date_col,
                        threshold_partial
                    )

        easy_duplicates.loc[pair_id, "type"] = duplicates_type

    return easy_duplicates


def combine_all_duplicates_one_model(
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        [easy_duplicates, subtle_duplicates],
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
