"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data, create_reduced_text_cols


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["wi_dataset",
                        "params:str_cols",
                        "params:cols_to_concatenate",
                        "params:concatenated_col_name",
                        "params:description_col",
                        "params:beginning_col_prefix",
                        "params:end_col_prefix",
                        "params:threshold_short_description"],
                outputs="preprocessed_dataset",
                name="preprocess_data_node",
                tags=[
                    'full',
                    'easy'
                     ]
            ),
            node(
                func=create_reduced_text_cols,
                inputs=["preprocessed_dataset",
                        "params:languages_list",
                        "params:description_col",
                        "params:reduced_col_prefix",
                        "params:very_reduced_col_prefix",
                        "params:proportion_words_to_filter_out"
                        ],
                outputs="preprocessed_dataset_with_extended_descriptions",
                name="create_reduced_descriptions_node"
            ),
            node(
                func=create_reduced_text_cols,
                inputs=["preprocessed_dataset_with_extended_descriptions",
                        "params:languages_list",
                        "params:concatenated_col_name",
                        "params:reduced_col_prefix",
                        "params:very_reduced_col_prefix",
                        "params:proportion_words_to_filter_out"
                        ],
                outputs="processed_dataset",
                name="create_reduced_texts_node"
            ),
        ],
        tags=[
            'best_model',
            'tf_idf',
            'multilingual_bert',
            'xlm_roberta'
            ]
    )
