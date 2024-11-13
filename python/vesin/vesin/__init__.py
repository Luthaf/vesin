import importlib.metadata

from ._ase import ase_neighbor_list  # noqa: F401
from ._neighbors import NeighborList  # noqa: F401


__version__ = importlib.metadata.version("vesin")
