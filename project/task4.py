from functools import reduce
from symtable import Symbol

from networkx.classes import MultiDiGraph
from scipy.sparse import bsr_matrix, vstack

from project.task2 import regex_to_dfa, graph_to_nfa
from project.task3 import AdjacencyMatrixFA


def _init_front(dfa: AdjacencyMatrixFA, nfa: AdjacencyMatrixFA):
    if len(dfa.start_states) != 1:
        raise RuntimeError("DFA should have only one start state")

    dfa_start_state = list(dfa.start_states)[0]
    return vstack(
        [
            bsr_matrix(
                ([True], ([dfa_start_state], [nfa_start_state])),
                shape=(dfa.states_count, nfa.states_count),
                dtype=bool,
            )
            for nfa_start_state in nfa.start_states
        ]
    )


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    permutation_matrices: dict[Symbol, bsr_matrix] = {}
    for symbol, matrix in dfa.bool_decomposition.items():
        permutation_matrices[symbol] = matrix.transpose()

    front = _init_front(dfa, nfa)
    visited = front
    while front.toarray().any():
        next_fronts: dict[Symbol, bsr_matrix] = {}
        for symbol in dfa.bool_decomposition.keys():
            if symbol not in nfa.bool_decomposition.keys():
                continue
            permutation_matrix = permutation_matrices[symbol]
            next_front = front @ nfa.bool_decomposition[symbol]
            next_fronts[symbol] = vstack(
                [
                    permutation_matrix
                    @ next_front[dfa.states_count * idx : dfa.states_count * (idx + 1)]
                    for idx in range(len(start_nodes))
                ]
            )
        front = reduce(lambda x, y: x + y, next_fronts.values(), front)
        front = front > visited
        visited = visited + front

    result: set[tuple[int, int]] = set()
    nfa_id_to_state_mapping = nfa.id_to_state_mapping
    for dfa_final in dfa.final_states:
        for nfa_start_idx, nfa_start in enumerate(nfa.start_states):
            visited_slice = visited[
                dfa.states_count * nfa_start_idx : dfa.states_count
                * (nfa_start_idx + 1)
            ]
            for nfa_reached in visited_slice.getrow(dfa_final).indices:
                if nfa_reached in nfa.final_states:
                    result.add(
                        (
                            nfa_id_to_state_mapping[nfa_start].value,
                            nfa_id_to_state_mapping[nfa_reached].value,
                        )
                    )
    return result
