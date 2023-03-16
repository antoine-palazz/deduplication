"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data, create_reduced_text_col


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
                        "params:threshold_short_description"],
                outputs="preprocessed_dataset",
                name="preprocess_data_node",
                tags=[
                    'full',
                    'easy',
                     ]
            ),
            node(
                func=create_reduced_text_col,
                inputs=["preprocessed_dataset",
                        "params:languages_list",
                        "params:concatenated_col_name",
                        "params:reduced_col_name"],
                outputs="processed_dataset",
                name="create_reduced_text_node"
            ),
        ],
        tags=[
            'best_model',
            'tf_idf',
            'multilingual_bert',
            'xlm_roberta'
            ]
    )
