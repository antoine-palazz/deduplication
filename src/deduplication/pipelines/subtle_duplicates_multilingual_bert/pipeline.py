"""
This is a boilerplate pipeline 'subtle_duplicates_multilingual_bert'
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
                outputs="tokens_multilingual_bert",
                name="tokenize_multilingual_bert_node"
            ),

            node(
                func=find_subtle_duplicates_from_tokens,
                inputs=["well_preprocessed_and_described_international_offers",
                        "tokens_multilingual_bert",
                        "params:str_cols",
                        "params:threshold_semantic_multilingual_bert",
                        "params:threshold_date",
                        "params:thresholds_similarity",
                        "params:thresholds_desc_len",
                        "params:hyperparameters",
                        "params:ner"
                        ],
                outputs="subtle_duplicates_multilingual_bert",
                name="identify_subtle_duplicates_multilingual_bert_node"
            )
        ],
        tags=[
            'multilingual_bert'
            ]
    )
