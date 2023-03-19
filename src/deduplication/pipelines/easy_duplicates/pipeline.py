"""
This is a boilerplate pipeline 'easy_duplicates'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import identify_exact_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_exact_duplicates,
                inputs=["preprocessed_complete_offers",
                        "params:list_cols_to_match_full",
                        "params:type_full",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:threshold_partial"],
                outputs="gross_full_duplicates",
                name="identify_gross_full_duplicates_node",
            ),
            node(
                func=identify_exact_duplicates,
                inputs=["preprocessed_complete_offers",
                        "params:list_cols_to_match_partial",
                        "params:type_partial",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:threshold_partial"],
                outputs="gross_partial_duplicates",
                name="identify_gross_partial_duplicates_node",
            ),
            node(
                func=identify_exact_duplicates,
                inputs=["extensively_preprocessed_described_offers",
                        "params:list_cols_to_match_semantic",
                        "params:type_semantic",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:threshold_partial"],
                outputs="gross_semantic_duplicates",
                name="identify_gross_semantic_duplicates_node",
            )
        ],
        tags=[
            'easy',
            'tf_idf',
            'multilingual_bert',
            'xlm_roberta',
            'final_models'
            ]
    )
