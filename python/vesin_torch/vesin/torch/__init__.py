import importlib.metadata

from ._c_lib import _load_library
from ._neighbors import NeighborList  # noqa: F401


__version__ = importlib.metadata.version("vesin-torch")

_load_library()
