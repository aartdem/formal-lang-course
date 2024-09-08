from dataclasses import dataclass
from typing import List

import cfpq_data
import networkx as nx


@dataclass(frozen=True)
class GraphInfo:
    node_count: int
    edge_count: int
    edge_labels_list: List[int]


def extract_graph_info(graph_name: str) -> GraphInfo:
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    return GraphInfo(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        edge_labels_list=cfpq_data.get_sorted_labels(graph),
    )


def save_two_cycles_graph_to_dot(
    size1: int, size2: int, label1: str, label2: str, path: str
):
    graph = cfpq_data.labeled_two_cycles_graph(
        n=size1, m=size2, labels=(label1, label2)
    )
    nx.nx_pydot.write_dot(graph, path)
