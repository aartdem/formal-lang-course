from functools import reduce
from symtable import Symbol

from networkx.classes import MultiDiGraph
from scipy.sparse import bsr_matrix, vstack

from project.task2 import regex_to_dfa, graph_to_nfa
from project.task3 import AdjacencyMatrixFA


def _create_helper_matrix(matrix: bsr_matrix) -> bsr_matrix:
    inv_rows = []
    inv_cols = []
    n, m = (matrix.shape[0], matrix.shape[1])
    if n != m:
        raise ValueError(f"Matrix ${matrix} should be square")
    for row_num in range(n):
        for col_num in matrix.getrow(row_num).indices:
            inv_rows.append(col_num)
            inv_cols.append(row_num)
    return bsr_matrix(
        ([True] * len(inv_rows), (inv_rows, inv_cols)), shape=(n, n), dtype=bool
    )


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
        permutation_matrices[symbol] = _create_helper_matrix(matrix)
    nfa_id_to_state = nfa.id_to_state_mapping

    front = _init_front(dfa, nfa)
    visited = front
    while front.toarray().any():
        next_fronts: dict[Symbol, bsr_matrix] = {}
        for symbol in dfa.bool_decomposition.keys():
            if symbol not in nfa.bool_decomposition:
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
    for dfa_idx in range(dfa.states_count):
        if dfa_idx in dfa.final_states:
            for nfa_start_idx, nfa_start in enumerate(start_nodes):
                visited_slice = visited[
                    nfa_start_idx * dfa.states_count : dfa.states_count
                    * (nfa_start_idx + 1)
                ]
                for nfa_final_idx in visited_slice.getrow(dfa_idx).indices:
                    nfa_final = nfa_id_to_state[nfa_final_idx].value
                    if nfa_final in final_nodes:
                        result.add((nfa_start, nfa_final))
    return result
