import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pyformlang
from networkx.classes import MultiDiGraph
from pyformlang.cfg import Variable, Terminal, Epsilon
from scipy.sparse import bsr_matrix

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    nfa = AdjacencyMatrixFA(graph_to_nfa(MultiDiGraph(graph), start_nodes, final_nodes))
    cfg_wnf = cfg_to_weak_normal_form(cfg)

    body_to_heads = defaultdict(list)
    for p in cfg_wnf.productions:
        body_to_heads[tuple(p.body)].append(p.head)

    bool_decomp: dict[Variable, bsr_matrix] = {
        head: bsr_matrix(np.zeros((nfa.states_count, nfa.states_count), dtype=bool))
        for head in list(itertools.chain.from_iterable(body_to_heads.values()))
    }

    def add_matrix_for_term(term: Terminal, matr: bsr_matrix):
        tup = tuple([term])
        if tup in body_to_heads:
            for h in body_to_heads[tup]:
                bool_decomp[h] += matr

    for symbol, matrix in nfa.bool_decomposition.items():
        add_matrix_for_term(Terminal(symbol), matrix)
    if "epsilon" not in nfa.bool_decomposition.keys():
        n = nfa.states_count
        add_matrix_for_term(
            Epsilon(),
            bsr_matrix(
                [[True if i == j else False for i in range(n)] for j in range(n)]
            ),
        )

    changed_vars = {var for var in bool_decomp.keys()}
    while changed_vars:
        current_var = changed_vars.pop()
        for body, heads in body_to_heads.items():
            if len(body) != 2 or current_var not in body:
                continue
            matrix_to_add = bool_decomp[body[0]] @ bool_decomp[body[1]]
            for head in heads:
                new_matrix = bool_decomp[head] + matrix_to_add
                if (bool_decomp[head] != new_matrix).count_nonzero() > 0:
                    bool_decomp[head] = new_matrix
                    changed_vars.add(head)

    return {
        (nfa.id_to_state[start_id].value, nfa.id_to_state[final_id].value)
        for start_id, final_id in zip(*bool_decomp[cfg_wnf.start_symbol].nonzero())
        if start_id in nfa.start_states_ids and final_id in nfa.final_states_ids
    }
