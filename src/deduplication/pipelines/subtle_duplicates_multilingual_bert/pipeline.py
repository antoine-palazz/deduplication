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
                inputs=["extensively_preprocessed_international_offers",
                        "params:concatenated_col_names",
                        "params:normal_description_type",
                        "params:dim_tokens",
                        "params:batch_size"
                        ],
                outputs="tokens_multilingual_bert",
                name="tokenize_multilingual_bert_node"
            ),

            node(
                func=find_subtle_duplicates_from_tokens,
                inputs=["extensively_preprocessed_international_offers",
                        "tokens_multilingual_bert",
                        "params:str_cols",
                        "params:cols_to_be_similar",
                        "params:title_col",
                        "params:description_col",
                        "params:date_col",
                        "params:id_col",
                        "params:language_col",
                        "params:threshold_similarity_multilingual",
                        "params:threshold_semantic_multilingual_bert",
                        "params:threshold_partial_multilingual",
                        "params:chunk_size"
                        ],
                outputs="subtle_duplicates_multilingual_bert",
                name="identify_subtle_duplicates_multilingual_bert_node"
            )
        ],
        tags=[
            'multilingual_bert'
            ]
    )
