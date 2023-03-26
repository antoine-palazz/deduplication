"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    filter_international_companies,
    filter_out_incomplete_offers,
    filter_out_poorly_described_offers,
    preprocess_data_basic,
    preprocess_data_extensive,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data_basic,
                inputs=["wi_dataset",
                        "params:str_cols",
                        "params:description_col",
                        "params:id_col",
                        "params:language_col"],
                outputs="preprocessed_dataset",
                name="basic_preprocessing_data_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["preprocessed_dataset",
                        "params:required_cols_full",
                        "params:nb_allowed_nans_full"],
                outputs="preprocessed_offers_for_full",
                name="filter_offers_for_full_node"
            ),

            node(
                func=preprocess_data_extensive,
                inputs=["preprocessed_dataset",
                        "params:str_cols",
                        "params:cols_to_concatenate",
                        "params:description_col",
                        "params:filtered_description_col",
                        "params:language_col",
                        "params:concatenated_col_names",
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
                inputs=["preprocessed_dataset",
                        "params:required_cols_partial",
                        "params:nb_allowed_nans_partial"],
                outputs="extensively_preprocessed_detailed_offers_for_partial",
                name="filter_detailed_offers_for_partial_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:required_cols_semantic",
                        "params:nb_allowed_nans_semantic"],
                outputs=(
                    "extensively_preprocessed_described_offers_for_semantic"
                ),
                name="filter_described_offers_for_semantic_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:required_cols_semantic_multilingual",
                        "params:nb_allowed_nans_semantic_multilingual"],
                outputs=(
                    "extensively_preprocessed_detailed_offers_for_semantic"
                ),
                name="filter_detailed_offers_for_semantic_multilingual_node"
            ),

            node(
                func=filter_international_companies,
                inputs=["extensively_preprocessed_dataset",
                        "params:cols_to_be_diversified_for_companies",
                        "params:company_col"],
                outputs="extensively_preprocessed_international_offers",
                name="filter_international_offers_node"
            ),
            node(
                func=filter_out_poorly_described_offers,
                inputs=["extensively_preprocessed_international_offers",
                        "params:cols_not_to_be_diversified_for_descriptions",
                        "params:description_col"],
                outputs="well_preprocessed_and_described_international_offers",
                name="filter_out_poorly_described_offers_node"
            )
        ],
        tags=[
            'easy',
            'tf_idf',
            'distiluse_multilingual',
            'multilingual_bert',
            'xlm_roberta',
            'final_models'
            ]
    )
