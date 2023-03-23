"""
This is a boilerplate pipeline 'subtle_duplicates_distiluse_multilingual'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import identify_subtle_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_subtle_duplicates,
                inputs=["extensively_preprocessed_dataset",
                        "params:concatenated_col_names",
                        "params:str_cols",
                        "params:cols_to_be_similar",
                        "params:normal_description_type",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params: language_col",
                        "params:dim_tokens",
                        "params:threshold_similarity_multilingual",
                        "params:threshold_semantic_distiluse_multilingual",
                        "params:threshold_partial_multilingual",
                        "params:chunk_size"
                        ],
                outputs="subtle_duplicates_distiluse_multilingual",
                name="identify_subtle_duplicates_distiluse_multilingual_node"
            )
        ],
        tags=[
            'distiluse_multilingual',
            'final_models'
            ]
    )
