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
                        "params:ner"],
                outputs="preprocessed_dataset",
                name="basic_preprocessing_data_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["preprocessed_dataset",
                        "params:FULL",
                        "params:required_cols_for_filtering",
                        "params:nb_allowed_nans_for_filtering"],
                outputs="preprocessed_offers_for_full",
                name="filter_offers_for_full_node"
            ),

            node(
                func=preprocess_data_extensive,
                inputs=["preprocessed_dataset",
                        "params:str_cols",
                        "params:cols_to_concatenate",
                        "params:languages_list",
                        "params:proportion_words_to_filter_out",
                        "params:threshold_short_text"],
                outputs="extensively_preprocessed_dataset",
                name="extensive_preprocessing_data_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:SEMANTIC",
                        "params:required_cols_for_filtering",
                        "params:nb_allowed_nans_for_filtering"],
                outputs=(
                    "very_preprocessed_described_offers_for_semantic"
                ),
                name="filter_described_offers_for_semantic_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:SEMANTIC_PARTIAL",
                        "params:required_cols_for_filtering",
                        "params:nb_allowed_nans_for_filtering"],
                outputs=(
                    "very_preprocessed_detailed_offers_for_semantic_partial"
                ),
                name="filter_detailed_offers_for_semantic_partial_node"
            ),
            node(
                func=filter_out_incomplete_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:SEMANTIC_MULTILINGUAL",
                        "params:required_cols_for_filtering",
                        "params:nb_allowed_nans_for_filtering"],
                outputs=(
                    "very_preprocessed_detailed_offers_for_semantic_lingual"
                ),
                name="filter_detailed_offers_for_semantic_multilingual_node"
            ),

            node(
                func=filter_out_poorly_described_offers,
                inputs=["extensively_preprocessed_dataset",
                        "params:cols_not_to_be_diversified_for_descriptions"],
                outputs="well_preprocessed_and_described_offers",
                name="filter_out_poorly_described_offers_node"
            ),
            node(
                func=filter_international_companies,
                inputs=["well_preprocessed_and_described_offers",
                        "params:cols_to_be_diversified_for_companies"],
                outputs="well_preprocessed_and_described_international_offers",
                name="filter_international_offers_node"
            )

        ],
        tags=[
            'easy',
            'tf_idf',
            'distiluse_multilingual',
            'multilingual_bert',
            'xlm_roberta',
            'final_models',
            'final_models_parallel_part'
            ]
    )
