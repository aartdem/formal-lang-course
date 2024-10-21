import json
from typing import List, Any


class IncorrectTestDataFileException(Exception):
    pass


def load_tests_from_json(path: str, test_name: str) -> List[Any]:
    with open(path) as f:
        d = json.load(f)
        if test_name not in d:
            raise IncorrectTestDataFileException(
                f"File '{path}' doesn't contain test data for test '{test_name}'"
            )
        else:
            return d[test_name]


def are_contain_same_elements(l1: list, l2: list) -> bool:
    if len(l1) != len(l2):
        return False
    for element in l1:
        if element not in l2:
            return False
    return True
