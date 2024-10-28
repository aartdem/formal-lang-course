import networkx as nx
import pyformlang
from networkx.classes import MultiDiGraph
from pyformlang.cfg import Production, Epsilon, CFG
from pyformlang.finite_automaton.finite_automaton import to_symbol

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    generates_eps = cfg.generate_epsilon()
    normal_form = cfg.to_normal_form()

    if generates_eps:
        start_symbol = normal_form.start_symbol
        return CFG(
            variables=normal_form.variables,
            terminals=normal_form.terminals | {Epsilon()},
            start_symbol=start_symbol,
            productions=normal_form.productions
            | {Production(start_symbol, [Epsilon()], False)},
        )
    else:
        return normal_form


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg_wnf = cfg_to_weak_normal_form(cfg)
    nfa = AdjacencyMatrixFA(graph_to_nfa(MultiDiGraph(graph), start_nodes, final_nodes))
    r = set()
    for p in cfg_wnf.productions:
        if len(p.body) == 2:
            continue
        terminal = p.body[0]
        if isinstance(terminal, Epsilon):
            for v in nfa.id_to_state.keys():
                r.add((p.head, v, v))
        else:
            symbol = to_symbol(terminal.value)
            if symbol in nfa.bool_decomposition.keys():
                for v, u in zip(*nfa.bool_decomposition[symbol].nonzero()):
                    r.add((p.head, v, u))

    new = r.copy()
    while new:
        N, u, v = new.pop()
        tuples_to_add = set()
        for M, w, x in r:
            if u == x:
                for p in cfg_wnf.productions:
                    tuple_to_add = (p.head, w, v)
                    if len(p.body) == 2 and p.body == [M, N] and tuple_to_add not in r:
                        tuples_to_add.add(tuple_to_add)
            if v == w:
                for p in cfg_wnf.productions:
                    tuple_to_add = (p.head, u, x)
                    if len(p.body) == 2 and p.body == [N, M] and tuple_to_add not in r:
                        tuples_to_add.add(tuple_to_add)
        r.update(tuples_to_add)
        new.update(tuples_to_add)

    answer = set()
    id_to_state = nfa.id_to_state
    for N, u, v in r:
        if (
            N == cfg_wnf.start_symbol
            and u in nfa.start_states_ids
            and v in nfa.final_states_ids
        ):
            answer.add((id_to_state[u].value, id_to_state[v].value))
    return answer
