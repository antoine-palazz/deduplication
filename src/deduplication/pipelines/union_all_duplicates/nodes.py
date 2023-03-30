"""
This is a boilerplate pipeline 'union_all_duplicates'
generated using Kedro 0.18.6
"""

import networkx as nx
import pandas as pd
from tqdm import tqdm

from deduplication.extras.utils import do_dates_differ_much


def add_transitivity_pairs_semantic(
    duplicates: pd.DataFrame,
    data: pd.DataFrame,
    threshold_date: dict
) -> pd.DataFrame:

    allowed_ids = set(data["id"])
    dates_arr = data['retrieval_date'].values
    indexes_from_id = dict(zip(data.id, data.index))

    # Creation of a non oriented graph representing the semantic pairs
    semantic_duplicates = duplicates[duplicates['type'].isin(['SEMANTIC', 'TEMPORAL'])]
    semantic_duplicates = semantic_duplicates[
        semantic_duplicates["id1"].isin(allowed_ids) &
        semantic_duplicates["id2"].isin(allowed_ids)
    ]
    G = nx.from_pandas_edgelist(semantic_duplicates, "id1", "id2")

    # Exploration of the connex components - Add edges to turn them into cliques
    dup_count_init = len(duplicates)
    print(f"{sum(1 for x in nx.connected_components(G))} different connex components")
    for subgraph in tqdm(nx.connected_components(G)):
        nodes = sorted(list(subgraph))
        len_nodes = len(nodes)
        if len_nodes == 2 or len_nodes > 80:
            continue

        for i in range(len_nodes-1):
            for j in range(i+1, len_nodes):
                if not G.has_edge(nodes[i], nodes[j]):

                    date_1 = pd.to_datetime(dates_arr[indexes_from_id[nodes[i]]])
                    date_2 = pd.to_datetime(dates_arr[indexes_from_id[nodes[j]]])
                    dates_differ = do_dates_differ_much(
                        date_1,
                        date_2,
                        threshold_date=threshold_date
                    )

                    if dates_differ == 'close_dates':
                        type_to_return = "SEMANTIC" if date_1 == date_2 else "TEMPORAL"
                        duplicates = duplicates.append({
                            "id1": nodes[i],
                            "id2": nodes[j],
                            "type": type_to_return
                        }, ignore_index=True)

    duplicates = duplicates.drop_duplicates(
        subset=["id1", "id2"]
    ).sort_values(
        by=["id1", "id2"],
        ignore_index=True
    )

    dup_count_end = len(duplicates)
    print(f"{dup_count_end - dup_count_init} new semantic pairs by transitivity")
    return duplicates


def add_transitivity_pairs_partial(
    duplicates: pd.DataFrame,
    data: pd.DataFrame
) -> pd.DataFrame:

    dup_count_init = len(duplicates)
    allowed_ids = set(data["id"])
    filtered_duplicates = duplicates[
        duplicates["id1"].isin(allowed_ids) &
        duplicates["id2"].isin(allowed_ids)
    ]

    semantic_duplicates = filtered_duplicates[filtered_duplicates['type'] == "SEMANTIC"]
    partial_duplicates = filtered_duplicates[filtered_duplicates['type'] == "PARTIAL"]

    G_semantic = nx.from_pandas_edgelist(semantic_duplicates, "id1", "id2")
    G_partial = nx.from_pandas_edgelist(partial_duplicates, "id1", "id2")
    G_semantic.add_nodes_from(G_partial)

    for index, row in tqdm(partial_duplicates.iterrows()):
        id1 = row['id1']
        id2 = row['id2']

        for node_i in G_semantic.subgraph(nx.shortest_path(G_semantic, id1)):
            for node_j in G_semantic.subgraph(nx.shortest_path(G_semantic, id2)):
                node_1 = min(node_i, node_j)
                node_2 = max(node_i, node_j)

                if node_1 != node_2 and not G_partial.has_edge(node_1, node_2):

                    G_partial.add_edge(node_1, node_2)
                    duplicates = duplicates.append({
                        "id1": node_1,
                        "id2": node_2,
                        "type": "PARTIAL"
                    }, ignore_index=True)

    duplicates = duplicates.sort_values(
        by=["type"],
    ).drop_duplicates(
        subset=["id1", "id2"]
    ).sort_values(
        by=["id1", "id2"],
        ignore_index=True
    )

    dup_count_end = len(duplicates)
    print(f"{dup_count_end - dup_count_init} new partial pairs by transitivity")
    return duplicates


def aggregate_duplicates_list(
    duplicates_list: list[pd.DataFrame],
    data: pd.DataFrame,
    threshold_date: dict
) -> pd.DataFrame:

    for duplicates in duplicates_list:
        print(f"Gross duplicates: {len(duplicates)}")

    all_duplicates = pd.concat(
        duplicates_list
    ).drop_duplicates(
        subset=['id1', 'id2']
    ).sort_values(
        by=['id1', 'id2']
    ).reset_index(drop=True)

    all_duplicates = add_transitivity_pairs_semantic(
        all_duplicates,
        data,
        threshold_date=threshold_date
    )
    # all_duplicates = add_transitivity_pairs_partial(all_duplicates, data)
    # Adds to many false positives for now

    return all_duplicates


def aggregate_easy_duplicates(
    gross_full_duplicates: pd.DataFrame,
    gross_partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    gross_semantic_multilingual_duplicates: pd.DataFrame,
    data: pd.DataFrame,
    threshold_date: dict
) -> pd.DataFrame:

    easy_duplicates = aggregate_duplicates_list(
        [gross_full_duplicates,
         gross_partial_duplicates,
         gross_semantic_duplicates,
         gross_semantic_multilingual_duplicates],
        data,
        threshold_date=threshold_date
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
    data: pd.DataFrame,
    threshold_date: dict
) -> pd.DataFrame:

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates, subtle_duplicates],
        data,
        threshold_date=threshold_date
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

    data = kwargs['well_preprocessed_and_described_offers']
    easy_duplicates = kwargs['easy']

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates] +
        [duplicates for name, duplicates in kwargs.items()
         if "duplicates" in name],
        data,
        threshold_date=kwargs['threshold_date']
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
