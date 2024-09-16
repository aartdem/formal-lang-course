from typing import Set, Any

from networkx import MultiDiGraph
from networkx.classes.reportviews import NodeView
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    EpsilonNFA,
    NondeterministicFiniteAutomaton,
)
from pyformlang.finite_automaton.finite_automaton import to_symbol
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    epsilon_nfa: EpsilonNFA = Regex(regex).to_epsilon_nfa()
    dfa: DeterministicFiniteAutomaton = epsilon_nfa.to_deterministic()
    return dfa.minimize()


def _to_int_if_possible(node: Any) -> Any:
    try:
        return int(node)
    except ValueError:
        return node


def _convert_nodes_to_states(nodeView: NodeView):
    return set(map(_to_int_if_possible, nodeView))


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nodes = _convert_nodes_to_states(graph.nodes)
    epsilon_nfa = EpsilonNFA(states=nodes)

    if not start_states:
        start_states = epsilon_nfa.states
    if not final_states:
        final_states = epsilon_nfa.states

    for state in start_states:
        if state not in epsilon_nfa.states:
            raise ValueError(
                f"start_states contains state '{state}' that is not in the graph"
            )
        epsilon_nfa.add_start_state(state)
    for state in final_states:
        if state not in epsilon_nfa.states:
            raise ValueError(
                f"final_states contains state '{state}' that is not in the graph"
            )
        epsilon_nfa.add_final_state(state)

    for edge in graph.edges(data=True):
        (node_from, node_to, edge_data) = edge
        node_from = _to_int_if_possible(node_from)
        node_to = _to_int_if_possible(node_to)

        if "label" in edge_data:
            symbol = edge_data["label"]
        else:
            symbol = "epsilon"

        epsilon_nfa.add_transition(node_from, to_symbol(symbol), node_to)

    return epsilon_nfa.remove_epsilon_transitions()
