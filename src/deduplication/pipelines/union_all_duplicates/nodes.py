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
    thresholds_dates: dict
) -> pd.DataFrame:
    """
    Being a semantic duplicate is a transitive property.
    If each offer is a node in a non oriented graph,
    with edges representing the "semantic duplicate" property,
    then this function transforms all of the connected components
    into cliques, ie applying this transitive property
    in order to find more semantic duplicates not caught earlier.

    Args:
        duplicates (pd.DataFrame): Dataset of the duplicated pairs found
        data (pd.DataFrame): Dataset of the offers
        thresholds_dates (dict): Thresholds for dates

    Returns:
        pd.DataFrame: New (bigger) dataset of duplicated pairs
    """
    allowed_ids = set(data["id"])  # Restrict the analysis to certain offers
    # Convert the dataset into a numpy array for faster computations
    dates_arr = data['retrieval_date'].values
    indexes_from_id = dict(zip(data.id, data.index))

    # Creation of a non oriented graph representing the semantic pairs
    semantic_duplicates = duplicates[
        duplicates['type'].isin(['SEMANTIC', 'TEMPORAL'])
    ]
    semantic_duplicates = semantic_duplicates[
        semantic_duplicates["id1"].isin(allowed_ids) &
        semantic_duplicates["id2"].isin(allowed_ids)
    ]
    G = nx.from_pandas_edgelist(semantic_duplicates, "id1", "id2")

    # Exploration of the connected components
    # Add edges to turn them into cliques
    dup_count_init = len(duplicates)
    print(
        f"{sum(1 for x in nx.connected_components(G))} connected components"
    )
    for subgraph in tqdm(nx.connected_components(G)):
        nodes = sorted(list(subgraph))
        len_nodes = len(nodes)
        if len_nodes == 2 or len_nodes > 80:
            continue  # Skip if the connected component is too big
            # Too long to compute and probably a mistake

        for i in range(len_nodes-1):
            for j in range(i+1, len_nodes):
                if not G.has_edge(nodes[i], nodes[j]):
                    # If an edge does not exist in the connected component

                    date_1 = pd.to_datetime(
                        dates_arr[indexes_from_id[nodes[i]]]
                    )
                    date_2 = pd.to_datetime(
                        dates_arr[indexes_from_id[nodes[j]]]
                    )
                    dates_differ = do_dates_differ_much(
                        date_1,
                        date_2,
                        thresholds_dates=thresholds_dates
                    )

                    if dates_differ == 'close_dates':
                        # We do not want to add edges between offers
                        # that are too far frome each other in time
                        type_to_return = ("SEMANTIC" if date_1 == date_2
                                          else "TEMPORAL")
                        new_dup = {"id1": nodes[i],
                                   "id2": nodes[j],
                                   "type": type_to_return}
                        duplicates = pd.concat(
                            [duplicates, pd.DataFrame([new_dup])],
                            ignore_index=True
                        )

    duplicates = duplicates.drop_duplicates(
        subset=["id1", "id2"]
    ).sort_values(
        by=["id1", "id2"],
        ignore_index=True
    )

    dup_count_end = len(duplicates)
    print(f"{dup_count_end - dup_count_init} new semantic by transitivity")
    return duplicates


def add_transitivity_pairs_partial(
    duplicates: pd.DataFrame,
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    If an offer A is a partial duplicate to an offer B, then:
    - B is a partial duplicate to all of the semantic duplicates of A
    - A is a partial duplicate to all of the semantic duplicates of B
    - Going even further, all of the semantic duplicates of A are
      partial duplicates to all of the semantic duplicates of B
    Using this property, we can catch many new partial duplicates
    that were not found before.
    However, this method requires to have a high level of confidence into
    the duplicates (whether they are partial or semantic) we already have

    Args:
        duplicates (pd.DataFrame): Dataset of the duplicated pairs found
        data (pd.DataFrame): Dataset of the offers

    Returns:
        pd.DataFrame: New (bigger) dataset of duplicated pairs
    """
    allowed_ids = set(data["id"])  # Restrict the analysis to certain offers
    # Convert the dataset into a numpy array for faster computations
    filtered_duplicates = duplicates[
        duplicates["id1"].isin(allowed_ids) &
        duplicates["id2"].isin(allowed_ids)
    ]

    semantic_duplicates = filtered_duplicates[
        filtered_duplicates['type'] == "SEMANTIC"
    ]
    partial_duplicates = filtered_duplicates[
        filtered_duplicates['type'] == "PARTIAL"
    ]

    # Creation of non oriented graphs representing semantic and partial pairs
    G_semantic = nx.from_pandas_edgelist(semantic_duplicates, "id1", "id2")
    G_partial = nx.from_pandas_edgelist(partial_duplicates, "id1", "id2")
    G_semantic.add_nodes_from(G_partial)

    dup_count_init = len(duplicates)
    for index, row in tqdm(partial_duplicates.iterrows()):
        # For all the partial duplicates
        id1 = row['id1']
        id2 = row['id2']

        for node_i in G_semantic.subgraph(
            nx.shortest_path(G_semantic, id1)
        ):  # All of the semantic duplicates of id1
            for node_j in G_semantic.subgraph(
                nx.shortest_path(G_semantic, id2)
            ):  # All of the semantic duplicates of id2

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
    print(f"{dup_count_end - dup_count_init} new partial by transitivity")
    return duplicates


def aggregate_duplicates_list(
    duplicates_list: list[pd.DataFrame],
    data: pd.DataFrame,
    thresholds_dates: dict
) -> pd.DataFrame:
    """
    Aggregates the duplicates from a list of duplicates tables
    all found using different approaches.
    Also looks for new duplicates using transitive properties.

    Caution: The list of duplicates tables shall be ordered from
    most reliable to least reliable, if applicable.

    Args:
        duplicates_list (list[pd.DataFrame]): List of duplicates tables
        data (pd.DataFrame): Dataset of offers
        thresholds_dates (dict): Thresholds for dates

    Returns:
        pd.DataFrame: Dataframe of duplicated pairs
    """
    for duplicates in duplicates_list:
        print(f"Gross duplicates: {len(duplicates)}")

    all_duplicates = pd.concat(
        duplicates_list
    ).drop_duplicates(
        subset=['id1', 'id2']
    ).sort_values(
        by=['id1', 'id2']
    ).reset_index(drop=True)

    # Add semantic pairs using transitivity
    all_duplicates = add_transitivity_pairs_semantic(
        all_duplicates,
        data,
        thresholds_dates=thresholds_dates
    )

    # Add partial pairs using transitivity of semantic
    # Not used for now as the current list of partials
    # is not reliable enough to be extended

    # all_duplicates = add_transitivity_pairs_partial(all_duplicates, data)

    return all_duplicates


def aggregate_easy_duplicates(
    gross_full_duplicates: pd.DataFrame,
    gross_partial_duplicates: pd.DataFrame,
    gross_semantic_duplicates: pd.DataFrame,
    gross_semantic_multilingual_duplicates: pd.DataFrame,
    data: pd.DataFrame,
    thresholds_dates: dict
) -> pd.DataFrame:
    """
    Aggregates all the duplicates from the "easy" approach,
    that was based on matches and mismatches of particular columns

    Args:
        gross_full_duplicates (pd.DataFrame)
        gross_partial_duplicates (pd.DataFrame)
        gross_semantic_duplicates (pd.DataFrame)
        gross_semantic_multilingual_duplicates (pd.DataFrame)
        data (pd.DataFrame): Dataset of offers
        thresholds_dates (dict): Thresholds for dates

    Returns:
        pd.DataFrame: Final list of "easily found" duplicates
    """
    easy_duplicates = aggregate_duplicates_list(
        [gross_full_duplicates,
         gross_partial_duplicates,
         gross_semantic_duplicates,
         gross_semantic_multilingual_duplicates],
        data,
        thresholds_dates=thresholds_dates
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
):
    """
    Prints the number of "new" duplicates found by a specific
    subtle approach compared to the previously found "easy" duplicates

    Args:
        easy_duplicates (pd.DataFrame): "Easily found" duplicates
        subtle_duplicates (pd.DataFrame): Dups found by complex approach
    """
    all_subtle_pairs = subtle_duplicates[['id1', 'id2']].merge(
        easy_duplicates[['id1', 'id2']],
        on=['id1', 'id2'],
        how='left',
        indicator=True
    )

    # Remove the "easy" duplicates to keep only the new ones
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
    thresholds_dates: dict
) -> pd.DataFrame:
    """
    Aggregates the "easy" duplicates and the "subtle" ones from one model

    Args:
        easy_duplicates (pd.DataFrame): "Easily found" duplicates
        subtle_duplicates (pd.DataFrame): Dups found by complex approach
        data (pd.DataFrame): Dataset of offers
        thresholds_dates (dict): Thresholds for dates

    Returns:
        pd.DataFrame: Aggregated table of duplicates
    """
    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates, subtle_duplicates],
        data,
        thresholds_dates=thresholds_dates
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
    """
    Aggregates the "easy" duplicates and all the "subtle" ones
    from different chosen models (as many as one wants)

    kwargs should contain:
    - "easy": The "easily found" duplicates
    - "xxx_duplicates": All the dups from more complex approaches to include
    - "well_preprocessed_and_described_offers": The dataset of offers
    - "thresholds_dates": The thresholds for dates

    Returns:
        pd.DataFrame: _description_
    """
    data = kwargs['well_preprocessed_and_described_offers']
    easy_duplicates = kwargs['easy']

    all_duplicates = aggregate_duplicates_list(
        [easy_duplicates] +
        [duplicates for name, duplicates in kwargs.items()
         if "duplicates" in name],
        data,
        thresholds_dates=kwargs['thresholds_dates']
    )

    if len(all_duplicates[
        all_duplicates['id1'] >= all_duplicates['id2']
    ]) > 0:
        print("PROBLEM: id1 >= id2 in the duplicates table")

    return all_duplicates


def describe_duplicates(all_duplicates: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the number of duplicates per type in the duplicates table

    Args:
        all_duplicates (pd.DataFrame): Dataframe of duplicated pairs

    Returns:
        pd.DataFrame: Number of duplicates per type
    """
    duplicates_description = all_duplicates.groupby('type').count(
    ).reset_index()
    return duplicates_description
