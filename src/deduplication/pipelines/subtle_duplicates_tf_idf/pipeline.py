"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
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
                        "params:max_df_tokenizer",
                        "params:chunk_size",
                        "params:threshold_semantic_tf_idf",
                        "params:threshold_partial_tf_idf"],
                outputs="subtle_duplicates_tf_idf",
                name="identify_subtle_duplicates_tf_idf"
            )
        ],
        tags=['tf_idf', 'best_model']
    )
