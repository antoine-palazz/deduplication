"""
This is a boilerplate pipeline 'subtle_duplicates_tf_idf'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from deduplication.extras.utils import find_subtle_duplicates_from_tokens

from .nodes import tokenize_texts


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=tokenize_texts,
                inputs=["extensively_preprocessed_dataset",
                        "params:concatenated_col_names",
                        "params:filtered_description_type",
                        "params:dim_tokens",
                        "params:max_df_tokenizer"
                        ],
                outputs="tokens_tf_idf",
                name="tokenize_tf_idf_node"
            ),

            node(
                func=find_subtle_duplicates_from_tokens,
                inputs=["extensively_preprocessed_dataset",
                        "tokens_tf_idf",
                        "params:str_cols",
                        "params:cols_to_be_similar",
                        "params:title_col",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:language_col",
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
