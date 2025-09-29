import networkx as nx
import pytest

from project.graph_utils import (
    GraphInfo,
    extract_graph_info,
    save_two_cycles_graph_to_dot,
)
from utils import load_tests_from_json

path_prefix = "tests/test_data/test_graph_utils"
test_extract_graph_info = "test_extract_graph_info"
test_save_two_cycles_graph_to_dot = "test_save_two_cycles_graph_to_dot"


@pytest.mark.parametrize(
    "test_data",
    load_tests_from_json(
        path=f"{path_prefix}/{test_extract_graph_info}.json",
        test_name=test_extract_graph_info,
    ),
)
def test_extract_graph_info(test_data: dict):
    graph_name = test_data["graph_name"]

    if test_data["is_graph_name_supported"]:
        expected_info = GraphInfo(
            node_count=int(test_data["node_count"]),
            edge_count=int(test_data["edge_count"]),
            edge_labels=set(test_data["edge_labels_list"]),
        )
        actual_info = extract_graph_info(graph_name)
        assert actual_info == expected_info
    else:
        with pytest.raises(FileNotFoundError):
            extract_graph_info(graph_name)


@pytest.mark.parametrize(
    "test_data",
    load_tests_from_json(
        path=f"{path_prefix}/{test_save_two_cycles_graph_to_dot}.json",
        test_name=test_save_two_cycles_graph_to_dot,
    ),
)
def test_save_two_cycles_graph_to_dot(test_data: dict, tmp_path):
    size1 = test_data["size1"]
    size2 = test_data["size2"]
    label1 = test_data["label1"]
    label2 = test_data["label2"]
    path = f"{tmp_path}/{test_data['path']}"

    if test_data["is_succeed"]:
        save_two_cycles_graph_to_dot(size1, size2, label1, label2, path)
        actual_graph = nx.DiGraph(nx.nx_pydot.read_dot(path))
        expected_graph = nx.DiGraph(
            list(
                map(
                    lambda edge: (edge[0], edge[1], edge[2]),
                    test_data["edges"],
                )
            )
        )

        assert nx.is_isomorphic(actual_graph, expected_graph, edge_match=dict.__eq__)

    else:
        with pytest.raises(ValueError):
            save_two_cycles_graph_to_dot(size1, size2, label1, label2, path)
