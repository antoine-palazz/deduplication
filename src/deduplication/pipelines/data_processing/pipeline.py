"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["wi_dataset",
                        "params:str_columns",
                        "params:concatenated_col_name"],
                outputs="preprocesssed_dataset",
                name="preprocess_data_node",
            ),
        ]
    )
