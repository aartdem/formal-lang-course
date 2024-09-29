from functools import reduce
from symtable import Symbol

import numpy as np
from networkx.classes import MultiDiGraph
from scipy.sparse import bsr_matrix

from project.task2 import regex_to_dfa, graph_to_nfa
from project.task3 import AdjacencyMatrixFA


def _calculate_helper_matrix(matrix: bsr_matrix) -> bsr_matrix:
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


def _init_front_and_visited(
    dfa: AdjacencyMatrixFA, nfa: AdjacencyMatrixFA
) -> (bsr_matrix, np.ndarray):
    if len(dfa.start_states) != 1:
        raise RuntimeError("DFA should have only one start state")

    dfa_start_state = list(dfa.start_states)[0]
    nfa_start_states = list(nfa.start_states)
    front = bsr_matrix(
        (
            [True] * len(nfa_start_states),
            ([dfa_start_state] * len(nfa_start_states), nfa_start_states),
        ),
        shape=(dfa.states_count, nfa.states_count),
        dtype=bool,
    )

    visited: np.array = np.zeros(len(nfa_start_states), dtype=bool)
    for start in nfa_start_states:
        visited[start] = True

    return front, visited


def _update_front_and_visited(
    last_front: bsr_matrix, visited: np.ndarray
) -> (bsr_matrix, np.ndarray):
    new_front = []
    for row in last_front.tocsr():
        new_row: np.ndarray = np.logical_and(row.toarray(), np.logical_not(visited))
        new_front.append(new_row)
    for row in new_front:
        visited = np.logical_or(visited, row)
    new_front = bsr_matrix(np.vstack(new_front))
    return new_front, visited

def _is_true_in_front(front: bsr_matrix) -> bool:
    for row in front.tocsr():
        if True in row.toarray():
            return True
    return False

def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    permutation_matrices: dict[Symbol, bsr_matrix] = {}
    for symbol, matrix in dfa.bool_decomposition.items():
        permutation_matrices[symbol] = _calculate_helper_matrix(matrix)

    front, visited = _init_front_and_visited(dfa, nfa)
    while _is_true_in_front(front):
        fronts: dict[Symbol, bsr_matrix] = {}
        for symbol in dfa.bool_decomposition.keys():
            if symbol not in nfa.bool_decomposition:
                continue
            nfa_adj_matrix = nfa.bool_decomposition[symbol]
            permutation_matrix = permutation_matrices[symbol]
            fronts[symbol] = permutation_matrix.dot(front.dot(nfa_adj_matrix))
        raw_front = reduce(lambda x, y: x + y, fronts.values(), front)
        front, visited = _update_front_and_visited(raw_front, visited)

    return set()
