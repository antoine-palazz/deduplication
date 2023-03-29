"""
This is a boilerplate pipeline 'subtle_duplicates_distiluse_multilingual'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from deduplication.extras.utils import find_subtle_duplicates_from_tokens

from .nodes import tokenize_texts


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=tokenize_texts,
                inputs=["well_preprocessed_and_described_international_offers",
                        "params:hyperparameters",
                        ],
                outputs="tokens_distiluse_multilingual",
                name="tokenize_distiluse_multilingual_node"
            ),

            node(
                func=find_subtle_duplicates_from_tokens,
                inputs=["well_preprocessed_and_described_international_offers",
                        "tokens_distiluse_multilingual",
                        "params:str_cols",
                        "params:threshold_semantic_distiluse_multilingual",
                        "params:threshold_date",
                        "params:thresholds_similarity",
                        "params:thresholds_desc_len",
                        "params:hyperparameters"
                        ],
                outputs="subtle_duplicates_distiluse_multilingual",
                name="identify_subtle_duplicates_distiluse_multilingual_node"
            )
        ],
        tags=[
            'distiluse_multilingual',
            'final_models',
            'final_models_sequential_part'
            ]
    )
