"""
This is a boilerplate pipeline 'subtle_duplicates_multilingual_bert'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import identify_subtle_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_subtle_duplicates,
                inputs=["processed_dataset",
                        "params:lemmatized_col_name",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:batch_size",
                        "params:chunk_size",
                        "params:threshold_semantic_xlm_roberta",
                        "params:threshold_partial_xlm_roberta"],
                outputs="subtle_duplicates_xlm_roberta",
                name="identify_subtle_duplicates_xlm_roberta_node"
            )
        ],
        tags=['xlm_roberta']
    )
