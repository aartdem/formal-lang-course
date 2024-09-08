import pytest

from utils import load_tests_from_json

path_prefix = "tests/test_data/test_graph_utils"
test_extract_graph_info = "test_extract_graph_info"


@pytest.mark.parametrize(
    "test_data",
    load_tests_from_json(
        path=f"{path_prefix}/{test_extract_graph_info}.json",
        test_name=test_extract_graph_info,
    ),
)
def test_extract_graph_info(test_data: dict):
    from project.graph_utils import extract_graph_info, GraphInfo

    graph_name = test_data["graph_name"]
    if test_data["is_graph_name_supported"]:
        assert True
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
