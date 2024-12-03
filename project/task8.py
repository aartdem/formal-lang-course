import networkx as nx
import pyformlang
from networkx.classes import MultiDiGraph
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    DeterministicFiniteAutomaton,
    NondeterministicTransitionFunction,
    State,
    Symbol,
)
from pyformlang.finite_automaton.finite_automaton import to_state

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def _to_adj_matrix(rsm: RecursiveAutomaton) -> AdjacencyMatrixFA:
    nfa_states = set()
    transition_func = NondeterministicTransitionFunction()
    start_states = set()
    final_states = set()
    for box_symbol, box in rsm.boxes.items():
        dfa: DeterministicFiniteAutomaton = box.dfa

        start_states.add(to_state((box_symbol, dfa.start_state)))

        for final_state in dfa.final_states:
            final_states.add(to_state((box_symbol, final_state)))

        for state in dfa.states:
            nfa_states.add(to_state((box_symbol, state)))

        for box_s_from, edges in dfa.to_dict().items():
            state_from = to_state((box_symbol, box_s_from))
            for symb_by, box_s_to in edges.items():
                if isinstance(box_s_to, State):
                    states_to = [to_state((box_symbol, box_s_to))]
                elif hasattr(box_s_to, "__iter__"):
                    states_to = [to_state((box_symbol, x)) for x in box_s_to]
                else:
                    raise ValueError("Unexpected format of DFA")
                for state_to in states_to:
                    transition_func.add_transition(state_from, symb_by, state_to)

    return AdjacencyMatrixFA(
        NondeterministicFiniteAutomaton(
            states=nfa_states,
            transition_function=transition_func,
            start_state=start_states,
            final_states=final_states,
        )
    )


def tensor_based_cfpq(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_matr = _to_adj_matrix(rsm)
    graph_matr = AdjacencyMatrixFA(
        graph_to_nfa(MultiDiGraph(graph), start_nodes, final_nodes)
    )

    changed = True
    while changed:
        new_transitions = _calculate_new_transitions(rsm_matr, graph_matr)
        changed = len(new_transitions) > 0
        for from_id, symbol, to_id in new_transitions:
            graph_matr.add_transition(from_id, symbol, to_id)

    result = set()
    if rsm.initial_label in graph_matr.bool_decomposition.keys():
        matr = graph_matr.bool_decomposition[rsm.initial_label]
        for i, j in zip(*matr.nonzero()):
            if i in graph_matr.start_states_ids and j in graph_matr.final_states_ids:
                result.add(
                    (graph_matr.id_to_state[i].value, graph_matr.id_to_state[j].value)
                )
    return result


def _calculate_new_transitions(
    rsm_matr: AdjacencyMatrixFA, graph_matr: AdjacencyMatrixFA
) -> list[tuple[int, Symbol, int]]:
    def unpack_intersect_state_id(
        intersect_state_id: int,
    ) -> tuple[Symbol, int, int]:
        state_value = intersection.id_to_state[intersect_state_id].value
        box_symbol = state_value[0].value[0]
        box_state_id = rsm_matr.state_to_id[state_value[0]]
        graph_state_id = graph_matr.state_to_id[state_value[1]]
        return box_symbol, box_state_id, graph_state_id

    new_transitions = []
    intersection = intersect_automata(rsm_matr, graph_matr)
    closure = intersection.get_transitive_closure()
    for closure_i, closure_j in zip(*closure.nonzero()):
        box_symb_i, rsm_st_i, graph_st_i = unpack_intersect_state_id(closure_i)
        box_symb_j, rsm_st_j, graph_st_j = unpack_intersect_state_id(closure_j)
        if box_symb_i != box_symb_j:
            raise ValueError(
                "The states of the rsm from different boxes cannot be reached from each other"
            )
        if (
            rsm_st_i in rsm_matr.start_states_ids
            and rsm_st_j in rsm_matr.final_states_ids
            and not graph_matr.has_transition(graph_st_i, box_symb_i, graph_st_j)
        ):
            new_transitions.append((graph_st_i, box_symb_i, graph_st_j))

    return new_transitions


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
