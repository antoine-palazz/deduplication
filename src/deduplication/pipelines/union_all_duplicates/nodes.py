"""
This is a boilerplate pipeline 'union_all_duplicates'
generated using Kedro 0.18.6
"""

import networkx as nx
import pandas as pd
from tqdm import tqdm


def add_transitivity_pairs_semantic(
    duplicates: pd.DataFrame,
    data: pd.DataFrame
) -> pd.DataFrame:

    allowed_ids = set(data["id"])
    dates_arr = data['retrieval_date'].values
    indexes_from_id = dict(zip(data.id, data.index))

    # Creation of a non oriented graph representing the semantic pairs
    semantic_duplicates = duplicates[duplicates['type'].isin(['SEMANTIC', 'TEMPORAL'])]
    semantic_duplicates = semantic_duplicates[
        semantic_duplicates["id1"].isin(allowed_ids)
    ]
    semantic_duplicates = semantic_duplicates[
        semantic_duplicates["id2"].isin(allowed_ids)
    ]
    G = nx.from_pandas_edgelist(semantic_duplicates, "id1", "id2")

    # Exploration of the connex components - Add edges to turn them into cliques
    dup_count_init = len(duplicates)
    print(f"{sum(1 for x in nx.connected_components(G))} different connex components")
    for subgraph in tqdm(nx.connected_components(G)):
        nodes = sorted(list(subgraph))
        len_nodes = len(nodes)
        if len_nodes == 2:
            continue

        for i in range(len_nodes-1):
            for j in range(i+1, len_nodes):
                if not G.has_edge(nodes[i], nodes[j]):

                    id1 = int(nodes[i])
                    id2 = int(nodes[j])
                    date_1 = dates_arr[indexes_from_id[id1]]
                    date_2 = dates_arr[indexes_from_id[id2]]
                    type_to_return = "SEMANTIC" if date_1 == date_2 else "TEMPORAL"

                    duplicates = duplicates.append({
                        "id1": id1,
                        "id2": id2,
                        "type": type_to_return
                    }, ignore_index=True)

    duplicates = duplicates.drop_duplicates(
        subset=["id1", "id2"]
    ).sort_values(
        by=["id1", "id2"],
        ignore_index=True
    )

    dup_count_end = len(duplicates)
    print(f"{dup_count_end - dup_count_init} new duplicates found by transitivity")
    return duplicates


def aggregate_duplicates_list(
    duplicates_list: list[pd.DataFrame],
    data: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = pd.concat(
        duplicates_list
    ).drop_duplicates(
        subset=['id1', 'id2']
    ).sort_values(
        by=['id1', 'id2']
    ).reset_index(drop=True)

    all_duplicates = add_transitivity_pairs_semantic(all_duplicates, data)

    return all_duplicates


def aggregate_easy_duplicates(
    gross_full_duplicates: pd.DataFrame,
    gross_partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    gross_semantic_multilingual_duplicates: pd.DataFrame,
    data: pd.DataFrame
) -> pd.DataFrame:

    easy_duplicates = aggregate_duplicates_list(
        [gross_full_duplicates,
         gross_partial_duplicates,
         gross_semantic_duplicates,
         gross_semantic_multilingual_duplicates],
        data
    )

    n_easy_duplicates = len(easy_duplicates)
    print(f'{n_easy_duplicates} "easy" duplicates isolated:')
    print(describe_duplicates(easy_duplicates))

    if len(easy_duplicates[
        easy_duplicates['id1'] >= easy_duplicates['id2']
            ]) > 0:
        print('PROBLEM: id1 >= id2 in the "easy" duplicates table')

    return easy_duplicates


def print_true_subtle_duplicates(
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame
) -> pd.DataFrame:

    all_subtle_pairs = subtle_duplicates[['id1', 'id2']].merge(
        easy_duplicates[['id1', 'id2']],
        on=['id1', 'id2'],
        how='left',
        indicator=True
    )

    true_subtles_duplicates = subtle_duplicates[
        ~all_subtle_pairs['_merge'].isin(['both'])
    ]

    print(
        f'{len(true_subtles_duplicates)} duplicates isolated with this model:'
    )
    print(describe_duplicates(true_subtles_duplicates))


def aggregate_all_duplicates_one_model(
    easy_duplicates: pd.DataFrame,
    subtle_duplicates: pd.DataFrame,
    data: pd.DataFrame
) -> pd.DataFrame:

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates, subtle_duplicates],
        data
    )

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
    ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    print_true_subtle_duplicates(easy_duplicates, subtle_duplicates)
    return all_duplicates


def aggregate_all_duplicates_several_models(
    **kwargs
) -> pd.DataFrame:

    data = kwargs['well_preprocessed_and_described_international_offers']
    easy_duplicates = kwargs['easy']

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates] +
        [duplicates for name, duplicates in kwargs.items()
         if name != "well_preprocessed_and_described_international_offers"],
        data
    )

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
    ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def describe_duplicates(all_duplicates: pd.DataFrame) -> pd.DataFrame:
    duplicates_description = all_duplicates.groupby('type').count(
    ).reset_index()
    return duplicates_description
