import importlib.metadata

from ._c_lib import register_ffi_targets as _register_ffi_targets
from ._neighbors import NeighborList  # noqa: F401


__version__ = importlib.metadata.version("vesin-jax")

_register_ffi_targets()
