import pyformlang
from pyformlang.cfg import Production, Epsilon, CFG


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
