"""Neighbor list implementations, one module per library.

Each implementation is a subclass of :class:`NeighborList` taking
``(atoms, cutoff, full_list, device)``, exposing a ``version`` class method and
a ``run`` method performing the neighbor list computation.
"""

from ._ase import ASENeighborList
from ._matscipy import MatscipyNeighborList
from ._mlipops import MLipopsNeighborList
from ._nnpops import NNPOpsNeighborList
from ._nvalchemi import NValchemiNeighborList
from ._pymatgen import PymatgenNeighborList
from ._sisl import SislNeighborList
from ._vesin import (
    VesinBruteForceNeighborList,
    VesinCellListNeighborList,
    VesinNeighborList,
    VesinSortedNeighborList,
    VesinVerletNeighborList,
    VesinVerletSortedNeighborList,
)


__all__ = [
    "ASENeighborList",
    "MLipopsNeighborList",
    "MatscipyNeighborList",
    "NNPOpsNeighborList",
    "NValchemiNeighborList",
    "PymatgenNeighborList",
    "SislNeighborList",
    "VesinBruteForceNeighborList",
    "VesinCellListNeighborList",
    "VesinNeighborList",
    "VesinSortedNeighborList",
    "VesinVerletNeighborList",
    "VesinVerletSortedNeighborList",
    "all_implementations",
    "default_implementations",
    "get_implementation",
]

_IMPLEMENTATIONS = {
    cls.name: cls
    for cls in (
        ASENeighborList,
        MatscipyNeighborList,
        PymatgenNeighborList,
        SislNeighborList,
        VesinNeighborList,
        VesinBruteForceNeighborList,
        VesinCellListNeighborList,
        VesinSortedNeighborList,
        VesinVerletNeighborList,
        VesinVerletSortedNeighborList,
        NValchemiNeighborList,
        NNPOpsNeighborList,
        MLipopsNeighborList,
    )
}


def get_implementation(name):
    """Return the class implementing the given neighbor list."""
    try:
        return _IMPLEMENTATIONS[name]
    except KeyError:
        available = ", ".join(_IMPLEMENTATIONS)
        raise ValueError(
            f"Unknown implementation '{name}', available implementations: {available}"
        ) from None


def all_implementations():
    """Return the names of all registered implementations."""
    return list(_IMPLEMENTATIONS)


def default_implementations():
    """Return the names of the implementations ``all`` expands to.

    This leaves out the implementations setting ``run_by_default = False``,
    which only run when named explicitly.
    """
    return [name for name, cls in _IMPLEMENTATIONS.items() if cls.run_by_default]
