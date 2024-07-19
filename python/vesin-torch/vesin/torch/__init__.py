import importlib.metadata

from ._c_lib import _load_library
from ._neighbors import NeighborList

__version__ = importlib.metadata.version("vesin-torch")
__all__ = ["NeighborList"]

_load_library()
