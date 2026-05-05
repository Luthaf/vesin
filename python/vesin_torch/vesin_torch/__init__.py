import importlib.metadata

from ._c_lib import load_library as _load_library
from ._compile import register_compile_support as _register_compile_support
from ._neighbors import NeighborList  # noqa: F401


__version__ = importlib.metadata.version("vesin-torch")

_load_library()
_register_compile_support()
