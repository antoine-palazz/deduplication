"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data, create_lemmatized_col


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["wi_dataset",
                        "params:str_cols",
                        "params:concatenated_col_name"],
                outputs="preprocessed_dataset",
                name="preprocess_data_node"
            ),
            node(
                func=create_lemmatized_col,
                inputs=["preprocessed_dataset",
                        "params:languages_list",
                        "params:concatenated_col_name",
                        "params:lemmatized_col_name"],
                outputs="processed_dataset",
                name="lemmatize_data_node"
            ),
        ],
        tags=[
            'full',
            'best_model',
            'tf_idf',
            'multilingual_bert',
            'xlm_roberta'
            ]
    )
