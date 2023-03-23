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
                inputs=["extensively_preprocessed_dataset",
                        "params:concatenated_col_names",
                        "params:str_cols",
                        "params:cols_to_be_similar",
                        "params:filtered_description_type",
                        "params:filtered_description_col",
                        "params:date_col",
                        "params:id_col",
                        "params: language_col",
                        "params:max_df_tokenizer",
                        "params:dim_tokens",
                        "params:threshold_similarity",
                        "params:threshold_semantic_tf_idf",
                        "params:threshold_partial",
                        "params:chunk_size"
                        ],
                outputs="subtle_duplicates_tf_idf",
                name="identify_subtle_duplicates_tf_idf_node"
            )
        ],
        tags=[
            'tf_idf',
            'final_models'
        ]
    )
