"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    combine_all_duplicates_one_model,
    combine_all_duplicates_from_best_models,
    describe_duplicates,
    differentiate_gross_semantic_duplicates
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=differentiate_gross_semantic_duplicates,
                inputs=["preprocessed_dataset",
                        "easy_gross_semantic_duplicates",
                        "full_duplicates",
                        "params:very_reduced_description_col_name",
                        "params:date_col",
                        "params:id_col",
                        "params:threshold_partial"],
                outputs="easy_duplicates",
                name="differentiate_gross_semantic_duplicates_node",
                tags=[
                    'easy',
                    'best_model',
                    'tf_idf',
                    'multilingual_bert',
                    'xlm_roberta'
                    ]
            ),
            node(
                func=describe_duplicates,
                inputs=["easy_duplicates"],
                outputs="easy_duplicates_description",
                name="describe_easy_duplicates_node",
                tags=[
                    'easy',
                    'best_model',
                    'tf_idf',
                    'multilingual_bert',
                    'xlm_roberta'
                    ]
            ),

            node(
                func=combine_all_duplicates_one_model,
                inputs=["full_duplicates",
                        "easy_duplicates",
                        "subtle_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf",
                name="combine_all_duplicates_tf_idf_node",
                tags=['tf_idf']
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf_description",
                name="describe_duplicates_tf_idf_node",
                tags=['tf_idf']
            ),

            node(
                func=combine_all_duplicates_one_model,
                inputs=["full_duplicates",
                        "easy_duplicates",
                        "subtle_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert",
                name="combine_all_duplicates_multilingual_bert_node",
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
                func=combine_all_duplicates_one_model,
                inputs=["full_duplicates",
                        "easy_duplicates",
                        "subtle_duplicates_xlm_roberta"],
                outputs="all_duplicates_xlm_roberta",
                name="combine_all_duplicates_xlm_roberta_node",
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
                func=combine_all_duplicates_from_best_models,
                inputs=["full_duplicates",
                        "easy_duplicates",
                        "params:best_model_temporal",
                        "params:best_model_partial",
                        "params:best_model_semantic",
                        "params:str_subtle_duplicates",
                        "params:project_path"],
                outputs="best_duplicates",
                name="combine_all_duplicates_from_best_models_node",
                tags=['best_model']
            ),
            node(
                func=describe_duplicates,
                inputs=["best_duplicates"],
                outputs="best_duplicates_description",
                name="describe_best_duplicates_node",
                tags=['best_model']
            ),
        ]
    )
