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
                inputs=["processed_dataset",
                        "params:cols_to_match",
                        "params:id_col"],
                outputs="full_duplicates",
                name="identify_full_duplicates"
            )
        ],
        tags=['full', 'tf_idf', 'multilingual_bert', 'xlm_roberta']
    )
