import typing as ty
from collections import UserDict


def to_int(key: ty.Union[str, int]) -> int:
    if isinstance(key, int):
        return key
    if not isinstance(key, str):
        raise RuntimeError(f"Unexpected key of type {type(key).__name__}: {key}")

    key = key.replace(" ", "")
    return int(key, 2)


class Counts(UserDict[int, float]):
    def __init__(self, counts: ty.Mapping[ty.Union[int, str], ty.Union[int, float]]):
        total = sum(counts.values())
        if total < 1e-10:
            print(counts)
        super().__init__({to_int(k): v / total for k, v in counts.items()})

    def __delitem__(self, key):
        super().__delitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"
