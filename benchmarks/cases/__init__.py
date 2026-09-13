from .base import BenchmarkCase
from .base import CaseStructure as CaseStructure
from .base import PairOptions as PairOptions
from .diamond import DiamondCase
from .graphite import GraphiteCase
from .nanodroplet import NanodropletCase
from .protein import ProteinCase
from .triclinic import TriclinicCase


_CASES = {
    cls.name: cls()
    for cls in (
        DiamondCase,
        GraphiteCase,
        NanodropletCase,
        ProteinCase,
        TriclinicCase,
    )
}


def all_cases() -> list[BenchmarkCase]:
    return list(_CASES.values())


def get_case(name: str) -> BenchmarkCase:
    try:
        return _CASES[name]
    except KeyError:
        available = ", ".join(_CASES)
        raise ValueError(
            f"Unknown benchmark case '{name}', available cases: {available}"
        ) from None
