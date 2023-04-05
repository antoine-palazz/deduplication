"""
This is a boilerplate pipeline 'subtle_duplicates_xlm_roberta'
generated using Kedro 0.18.6
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
                        "params:hyperparameters"
                        ],
                outputs="tokens_xlm_roberta",
                name="tokenize_xlm_roberta_node"
            ),

            node(
                func=find_subtle_duplicates_from_tokens,
                inputs=["well_preprocessed_and_described_international_offers",
                        "tokens_xlm_roberta",
                        "params:str_cols",
                        "params:threshold_semantic_xlm_roberta",
                        "params:thresholds_dates",
                        "params:thresholds_similarity",
                        "params:thresholds_desc_len",
                        "params:hyperparameters",
                        "params:ner"
                        ],
                outputs="subtle_duplicates_xlm_roberta",
                name="identify_subtle_duplicates_xlm_roberta_node"
            )
        ],
        tags=[
            'xlm_roberta'
            ]
    )
