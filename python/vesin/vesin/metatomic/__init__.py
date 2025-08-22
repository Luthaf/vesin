import metatomic.torch
import torch


# Dynamically check the dependencies versions since we can not declare them
# as dependencies in pyproject.toml (we only want to provide code in this module
# if the user already has the correct dependencies installed).
torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
if torch_version < (2, 3):
    raise ImportError(
        f"Found torch v{torch.__version__}, but vesin.metatomic requires torch >= 2.3"
    )

mta_version = tuple(map(int, metatomic.torch.__version__.split(".")[:3]))
if mta_version < (0, 1, 3) or mta_version >= (0, 2, 0):
    raise ImportError(
        f"Found metatomic.torch v{torch.__version__}, "
        "but vesin.metatomic requires metatomic.torch >=0.1.3,<0.2"
    )


from ._model import compute_requested_neighbors  # noqa: E402
from ._neighbors import NeighborList  # noqa: E402


__all__ = ["NeighborList", "compute_requested_neighbors"]
