"""
This is a boilerplate pipeline 'subtle_duplicates_xlm_roberta'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import identify_subtle_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_subtle_duplicates,
                inputs=["extensively_preprocessed_dataset",
                        "params:concatenated_col_name",
                        "params:str_cols",
                        "params:cols_to_be_similar",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:dim_tokens",
                        "params:threshold_similarity_multilingual",
                        "params:threshold_semantic_xlm_roberta",
                        "params:threshold_partial_multilingual",
                        "params:batch_size",
                        "params:chunk_size"
                        ],
                outputs="subtle_duplicates_xlm_roberta",
                name="identify_subtle_duplicates_xlm_roberta_node"
            )
        ],
        tags=['xlm_roberta']
    )
