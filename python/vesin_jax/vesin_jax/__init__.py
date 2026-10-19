import importlib.metadata

from ._neighbors import NeighborList  # noqa: F401


__version__ = importlib.metadata.version("vesin-jax")
