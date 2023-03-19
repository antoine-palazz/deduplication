"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_data_basic,
    filter_out_incomplete_offers,
    preprocess_data_extensive
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data_basic,
                inputs=["wi_dataset",
                        "params:str_cols"],
                outputs="preprocessed_dataset",
                name="basic_preprocessing_data_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["preprocessed_dataset",
                        "params:required_cols_full",
                        "params:nb_allowed_nans_full"],
                outputs="preprocessed_complete_offers",
                name="filter_out_incomplete_offers_for_full_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["preprocessed_dataset",
                        "params:required_cols_partial",
                        "params:nb_allowed_nans_partial"],
                outputs="preprocessed_quasi_complete_offers",
                name="filter_out_incomplete_offers_for_partial_node"
            ),

            node(
                func=preprocess_data_extensive,
                inputs=["preprocessed_dataset",
                        "params:str_cols",
                        "params:description_col",
                        "params:language_col",
                        "params:cols_to_concatenate",
                        "params:concatenated_col_name",
                        "params:languages_list",
                        "params:beginning_prefix",
                        "params:end_prefix",
                        "params:proportion_words_to_filter_out",
                        "params:threshold_short_text"],
                outputs="extensively_preprocessed_dataset",
                name="extensive_preprocessing_data_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:required_cols_semantic",
                        "params:nb_allowed_nans_semantic"],
                outputs="extensively_preprocessed_described_offers",
                name="filter_out_undescribed_offers_for_semantic_node"
            )
        ],
        tags=[
            'easy',
            'tf_idf',
            'multilingual_bert',
            'xlm_roberta',
            'final_models'
            ]
    )
