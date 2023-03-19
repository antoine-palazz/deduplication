"""
This is a boilerplate pipeline 'union_all_duplicates'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    aggregate_easy_duplicates,
    aggregate_all_duplicates_one_model,
    aggregate_all_duplicates_several_models,
    describe_duplicates
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aggregate_easy_duplicates,
                inputs=["gross_full_duplicates",
                        "gross_partial_duplicates",
                        "gross_semantic_duplicates"],
                outputs="easy_duplicates",
                name="aggregate_easy_duplicates_node",
                tags=[
                    'easy',
                    'tf_idf',
                    'multilingual_bert',
                    'xlm_roberta',
                    'final_models'
                     ]
            ),
            node(
                func=describe_duplicates,
                inputs=["easy_duplicates"],
                outputs="easy_duplicates_description",
                name="describe_easy_duplicates_node",
                tags=[
                    'easy',
                    'tf_idf',
                    'multilingual_bert',
                    'xlm_roberta',
                    'final_models'
                     ]
            ),

            node(
                func=aggregate_all_duplicates_one_model,
                inputs=["easy_duplicates",
                        "subtle_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf",
                name="aggregate_all_duplicates_tf_idf_node",
                tags=[
                    'tf_idf',
                    'final_models'
                     ]
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf_description",
                name="describe_duplicates_tf_idf_node",
                tags=[
                    'tf_idf',
                    'final_models'
                     ]
            ),

            node(
                func=aggregate_all_duplicates_one_model,
                inputs=["easy_duplicates",
                        "subtle_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert",
                name="aggregate_all_duplicates_multilingual_bert_node",
                tags=['multilingual_bert']
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert_description",
                name="describe_duplicates_multilingual_bert_node",
                tags=['multilingual_bert']
            ),

            node(
                func=aggregate_all_duplicates_one_model,
                inputs=["easy_duplicates",
                        "subtle_duplicates_xlm_roberta"],
                outputs="all_duplicates_xlm_roberta",
                name="aggregate_all_duplicates_xlm_roberta_node",
                tags=['xlm_roberta']
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_xlm_roberta"],
                outputs="all_duplicates_xlm_roberta_description",
                name="describe_duplicates_xlm_roberta_node",
                tags=['xlm_roberta']
            ),

            node(
                func=aggregate_all_duplicates_several_models,
                inputs=["easy_duplicates",
                        "params:final_duplicates_str_list",
                        "params:project_path"],
                outputs="best_duplicates",
                name="aggregate_all_duplicates_from_best_models_node",
                tags=['final_models']
            ),
            node(
                func=describe_duplicates,
                inputs=["best_duplicates"],
                outputs="best_duplicates_description",
                name="describe_best_duplicates_node",
                tags=['final_models']
            )
        ]
    )