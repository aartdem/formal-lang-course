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
    bool_decomp: dict[Variable, bsr_matrix] = {
        p.head: bsr_matrix(np.zeros((nfa.states_count, nfa.states_count), dtype=bool))
        for p in cfg_wnf.productions
    }

    body_to_heads = defaultdict(list)
    for p in cfg_wnf.productions:
        body_to_heads[tuple(p.body)].append(p.head)

    def process_terminal(term: Terminal, matr: bsr_matrix):
        tup = tuple([term])
        for h in body_to_heads[tup]:
            bool_decomp[h] = bool_decomp[h] + matr

    for symbol, matrix in nfa.bool_decomposition.items():
        process_terminal(Terminal(symbol), matrix)
    if "epsilon" not in nfa.bool_decomposition.keys():
        n = nfa.states_count
        process_terminal(
            Epsilon(),
            bsr_matrix(
                ([True] * n, (list(range(n)), list(range(n)))),
                shape=(n, n),
                dtype=bool,
            ),
        )

    changed = True
    while changed:
        changed = False
        for body, heads in body_to_heads.items():
            if len(body) != 2:
                continue
            new_matrix = bool_decomp[body[0]] @ bool_decomp[body[1]]
            for head in heads:
                before = bool_decomp[head]
                bool_decomp[head] = bool_decomp[head] + new_matrix
                if not changed:
                    changed = len((before != bool_decomp[head]).nonzero()[0]) != 0

    return {
        (nfa.id_to_state[start_id].value, nfa.id_to_state[final_id].value)
        for start_id, final_id in zip(*bool_decomp[cfg_wnf.start_symbol].nonzero())
        if start_id in nfa.start_states_ids and final_id in nfa.final_states_ids
    }
