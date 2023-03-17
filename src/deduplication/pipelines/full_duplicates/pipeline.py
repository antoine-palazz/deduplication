"""
This is a boilerplate pipeline 'full_duplicates'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import identify_full_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_full_duplicates,
                inputs=["preprocessed_dataset",
                        "params:full_type",
                        "params:list_cols_to_match_full",
                        "params:backup_cols_to_match_full",
                        "params:id_col"],
                outputs="full_duplicates",
                name="identify_full_duplicates_node",
                tags=['full']
            ),
            node(
                func=identify_full_duplicates,
                inputs=["preprocessed_dataset_with_extended_descriptions",
                        "params:semantic_type",
                        "params:list_cols_to_match_gross_semantic",
                        "params:backup_cols_to_match_gross_semantic",
                        "params:id_col"],
                outputs="easy_gross_semantic_duplicates",
                name="identify_easy_gross_semantic_duplicates_node"
            )
        ],
        tags=[
            'easy',
            'best_model',
            'tf_idf',
            'multilingual_bert',
            'xlm_roberta'
            ]
    )
