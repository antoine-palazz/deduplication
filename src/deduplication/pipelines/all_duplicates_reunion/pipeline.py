"""
This is a boilerplate pipeline 'all_duplicates_reunion'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import combine_all_duplicates, describe_duplicates


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=combine_all_duplicates,
                inputs=["full_duplicates",
                        "subtle_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf",
                name="combine_all_duplicates_tf_idf"
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_tf_idf"],
                outputs="all_duplicates_tf_idf_description",
                name="describe_duplicates_tf_idf"
            ),
            node(
                func=combine_all_duplicates,
                inputs=["full_duplicates",
                        "subtle_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert",
                name="combine_all_duplicates_multilingual_bert"
            ),
            node(
                func=describe_duplicates,
                inputs=["all_duplicates_multilingual_bert"],
                outputs="all_duplicates_multilingual_bert_description",
                name="describe_duplicates_multilingual_bert"
            )
        ]
    )
