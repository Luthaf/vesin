from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ase import Atoms


@dataclass
class CaseStructure:
    """A single structure produced by a benchmark case."""

    label: str
    build: Callable[[], Atoms]


@dataclass(frozen=True)
class PairOptions:
    """A pair generation option (cutoff and full_list flag)."""

    cutoff: float
    full_list: bool


class BenchmarkCase(ABC):
    """A family of structures to benchmark against."""

    name: str = ""

    options: list[PairOptions]

    @abstractmethod
    def structures(self) -> list[CaseStructure]:
        """Return the structures to benchmark, in order."""
