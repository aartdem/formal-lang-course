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
    cfg_weak_normal_form = cfg_to_weak_normal_form(cfg)
    nfa = AdjacencyMatrixFA(graph_to_nfa(MultiDiGraph(graph), start_nodes, final_nodes))
    r = set()
    for p in cfg_weak_normal_form.productions:
        if len(p.body) == 2:
            continue
        terminal = p.body[0]
        if isinstance(terminal, Epsilon):
            for node_v in nfa.id_to_state.keys():
                r.add((p.head, node_v, node_v))
        else:
            symbol = to_symbol(terminal.value)
            if symbol in nfa.bool_decomposition.keys():
                for row, col in zip(*nfa.bool_decomposition[symbol].nonzero()):
                    r.add((p.head, row, col))

    new = r.copy()
    while new:
        (var_n, node_u, node_v) = new.pop()
        tuples_to_add = set()
        for (
            var_m,
            node_w,
            node_x,
        ) in r:
            if node_u == node_x:
                for p in cfg_weak_normal_form.productions:
                    tuple_to_add = (p.head, node_w, node_v)
                    if (
                        len(p.body) == 2
                        and p.body == [var_m, var_n]
                        and tuple_to_add not in r
                    ):
                        tuples_to_add.add(tuple_to_add)
            elif node_v == node_w:
                for p in cfg_weak_normal_form.productions:
                    tuple_to_add = (p.head, node_u, node_x)
                    if (
                        len(p.body) == 2
                        and p.body == [var_n, var_m]
                        and tuple_to_add not in r
                    ):
                        tuples_to_add.add(tuple_to_add)
        r.update(tuples_to_add)
        new.update(tuples_to_add)

    answer = set()
    id_to_state = nfa.id_to_state
    for var, node_u, node_v in r:
        if (
            var == cfg_weak_normal_form.start_symbol
            and node_u in nfa.start_states_ids
            and node_v in nfa.final_states_ids
        ):
            answer.add((id_to_state[node_u].value, id_to_state[node_v].value))
    return answer
