import cfpq_data
import networkx as nx
import pytest
from networkx import MultiDiGraph
from pyformlang.regular_expression import MisformedRegexError

from project.graph_utils import extract_graph_info, save_two_cycles_graph_to_dot
from project.task2 import regex_to_dfa, graph_to_nfa


class TestRegexToDfaAdditional:
    def test_regex_to_dfa_is_empty(self):
        dfa = regex_to_dfa("")
        assert dfa.is_empty()

    def test_regex_to_dfa_incorrect_regex(self):
        with pytest.raises(MisformedRegexError):
            regex_to_dfa("( a")


class TestGraphToNfaAdditional:
    graph_for_test = MultiDiGraph()
    graph_for_test.add_nodes_from([0, 1, 2])
    graph_for_test.add_edge(0, 1, "a")
    graph_for_test.add_edge(1, 1, "b")
    graph_for_test.add_edge(1, 2, "c")

    def test_graph_to_nfa_incorrect_start_states(self):
        with pytest.raises(ValueError):
            graph_to_nfa(self.graph_for_test, {-1}, set())

    def test_graph_to_nfa_incorrect_end_states(self):
        with pytest.raises(ValueError):
            graph_to_nfa(self.graph_for_test, set(), {4})

    def test_graph_to_nfa_with_import_by_name(self):
        graph_name = "generations"
        graph_info = extract_graph_info(graph_name)
        graph_path = cfpq_data.download(graph_name)
        graph = cfpq_data.graph_from_csv(graph_path)

        nfa = graph_to_nfa(graph, {0}, {1})
        assert len(nfa.states) == graph_info.node_count
        assert nfa.get_number_transitions() == graph_info.edge_count
        assert nfa.symbols == graph_info.edge_labels
        assert nfa.start_states == {0}
        assert nfa.final_states == {1}

    def test_graph_to_nfa_with_create(self, tmp_path):
        n = 3
        m = 4
        graph_path = f"{tmp_path}/test_graph"
        save_two_cycles_graph_to_dot(n, m, "a", "b", graph_path)
        graph = nx.nx_pydot.read_dot(graph_path)

        nfa = graph_to_nfa(graph, {0}, {n + 1})
        assert len(nfa.states) == n + m + 1
        assert nfa.get_number_transitions() == n + m + 2
        assert nfa.symbols == {"a", "b"}
        assert nfa.start_states == {0}
        assert nfa.final_states == {4}
