import importlib.metadata
import os

import torch

from ._c_lib import _load_library

__version__ = importlib.metadata.version("vesin-torch")


if os.environ.get("VESIN_IMPORT_FOR_SPHINX", "0") != "0":
    from .documentation import NeighborList
else:
    _load_library()
    NeighborList = torch.classes.vesin.NeighborList


__all__ = ["NeighborList"]
