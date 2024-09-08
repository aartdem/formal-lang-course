import json
from typing import List, Any


class IncorrectDataFileException(Exception):
    pass


def load_tests_from_json(path: str, test_name: str) -> List[Any]:
    with open(path) as f:
        d = json.load(f)
        if test_name not in d:
            raise IncorrectDataFileException(
                f"File '{path}' doesn't contain test data for test '{test_name}'"
            )
        else:
            return d[test_name]
