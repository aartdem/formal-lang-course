import pytest
from pyformlang.cfg import CFG, Variable

from project.task6 import cfg_to_weak_normal_form
from tests.utils import load_tests_from_json, are_contain_same_elements

path = "tests/test_data/test_cfg_to_weak_normal_form.json"
test_cfg_to_weak_normal_form = "test_cfg_to_weak_normal_form"


@pytest.mark.parametrize(
    "test_data", load_tests_from_json(path=path, test_name=test_cfg_to_weak_normal_form)
)
def test_cfg_to_weak_normal_form(test_data: dict):
    cfg = CFG.from_text(test_data["text"])
    weak_normal_form = cfg_to_weak_normal_form(cfg)

    assert is_weak_normal_form(weak_normal_form)
    assert cfg.generate_epsilon() == weak_normal_form.generate_epsilon()
    assert are_contain_same_elements(
        list(cfg.get_words(10)), list(weak_normal_form.get_words(10))
    )


def is_weak_normal_form(cfg: CFG):
    for production in cfg.productions:
        if len(production.body) == 1:
            symbol = production.body[0]
            if isinstance(symbol, Variable):
                return False

        elif len(production.body) == 2:
            symbol1 = production.body[0]
            symbol2 = production.body[0]
            if not isinstance(symbol1, Variable) or not isinstance(symbol2, Variable):
                return False
        else:
            return False

    return True
