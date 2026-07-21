import importlib.metadata

from ._ase import ase_neighbor_list
from ._c_lib import get_library
from ._neighbors import NeighborList


__version__ = importlib.metadata.version("vesin")

__all__ = ["ase_neighbor_list", "NeighborList"]


get_library()
