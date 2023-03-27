"""
This is a boilerplate pipeline 'easy_duplicates'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import identify_exact_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identify_exact_duplicates,
                inputs=["preprocessed_offers_for_full",
                        "params:list_cols_to_match",
                        "params:list_cols_to_mismatch",
                        "params:FULL",
                        "params:str_cols",
                        "params:threshold_date",
                        "params:thresholds_similarity",
                        "params:thresholds_desc_len"],
                outputs="gross_full_duplicates",
                name="identify_gross_full_duplicates_node"
            ),
            node(
                func=identify_exact_duplicates,
                inputs=["extensively_preprocessed_detailed_offers_for_partial",
                        "params:list_cols_to_match",
                        "params:list_cols_to_mismatch",
                        "params:PARTIAL",
                        "params:str_cols",
                        "params:threshold_date",
                        "params:thresholds_similarity",
                        "params:thresholds_desc_len"],
                outputs="gross_partial_duplicates",
                name="identify_gross_partial_duplicates_node"
            ),
            node(
                func=identify_exact_duplicates,
                inputs=[
                    "extensively_preprocessed_described_offers_for_semantic",
                    "params:list_cols_to_match",
                    "params:list_cols_to_mismatch",
                    "params:SEMANTIC",
                    "params:str_cols",
                    "params:threshold_date",
                    "params:thresholds_similarity",
                    "params:thresholds_desc_len"
                    ],
                outputs="gross_semantic_duplicates",
                name="identify_gross_semantic_duplicates_node"
            ),
            node(
                func=identify_exact_duplicates,
                inputs=[
                    "extensively_preprocessed_detailed_offers_for_semantic",
                    "params:list_cols_to_match",
                    "params:list_cols_to_mismatch",
                    "params:SEMANTIC_MULTILINGUAL",
                    "params:str_cols",
                    "params:threshold_date",
                    "params:thresholds_similarity",
                    "params:thresholds_desc_len"
                ],
                outputs="gross_semantic_multilingual_duplicates",
                name="identify_gross_semantic_multilingual_duplicates_node"
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
