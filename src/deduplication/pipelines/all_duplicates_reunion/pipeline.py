"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    combine_all_duplicates_one_model,
    combine_all_duplicates_from_best_models,
    describe_duplicates
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=combine_all_duplicates_one_model,
                inputs=["full_duplicates",
                        "subtle_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf",
                name="combine_all_duplicates_tf_idf",
                tags=['tf_idf']
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf_description",
                name="describe_duplicates_tf_idf",
                tags=['tf_idf']
            ),

            node(
                func=combine_all_duplicates_one_model,
                inputs=["full_duplicates",
                        "subtle_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert",
                name="combine_all_duplicates_multilingual_bert",
                tags=['multilingual_bert']
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert_description",
                name="describe_duplicates_multilingual_bert",
                tags=['multilingual_bert']
            ),

            node(
                func=combine_all_duplicates_one_model,
                inputs=["full_duplicates",
                        "subtle_duplicates_xlm_roberta"],
                outputs="all_duplicates_xlm_roberta",
                name="combine_all_duplicates_xlm_roberta",
                tags=['xlm_roberta']
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_xlm_roberta"],
                outputs="all_duplicates_xlm_roberta_description",
                name="describe_duplicates_xlm_roberta",
                tags=['xlm_roberta']
            ),

            node(
                func=combine_all_duplicates_from_best_models,
                inputs=["full_duplicates",
                        "params:best_subtle_duplicates_temporal",
                        "params:best_subtle_duplicates_partial",
                        "params:best_subtle_duplicates_semantic"],
                outputs="best_duplicates",
                name="combine_all_duplicates_from_best_models",
                tags=['best_model']
            ),
            node(
                func=describe_duplicates,
                inputs=["best_duplicates"],
                outputs="best_duplicates_description",
                name="describe_best_duplicates",
                tags=['best_model']
            ),
        ]
    )
