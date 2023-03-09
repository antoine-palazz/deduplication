"""
This is a boilerplate pipeline 'subtle_duplicates_multilingual_bert'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import identify_subtle_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_subtle_duplicates,
                inputs=["preprocessed_dataset",
                        "params:languages_list",
                        "params:concatenated_col_name",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:max_df_tokenizer",
                        "params:chunk_size",
                        "params:threshold_semantic",
                        "params:threshold_partial"],
                outputs="subtle_duplicates_multilingual_bert",
                name="identify_subtle_duplicates_multilingual_bert"
            )
        ]
    )
