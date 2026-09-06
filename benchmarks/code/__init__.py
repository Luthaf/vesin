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
from ._vesin import VesinNeighborList


__all__ = [
    "ASENeighborList",
    "MLipopsNeighborList",
    "MatscipyNeighborList",
    "NNPOpsNeighborList",
    "NValchemiNeighborList",
    "PymatgenNeighborList",
    "SislNeighborList",
    "VesinNeighborList",
    "all_implementations",
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
        raise ValueError(f"Unknown implementation: {name}") from None


def all_implementations():
    """Return the names of all registered implementations."""
    return list(_IMPLEMENTATIONS)
