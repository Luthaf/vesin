import re
import warnings

import metatomic.torch


def _check_mta_version():
    # Dynamically check the dependencies versions since we can not declare them
    # as dependencies in pyproject.toml (we only want to provide code in this module
    # if the user already has the correct dependencies installed).
    version_re = re.compile(r"(\d+)\.(\d+)\.(\d+)(.*)?")
    groups = version_re.match(metatomic.torch.__version__).groups()

    mta_version = tuple(map(int, groups[:3]))
    is_dev = groups[3].startswith(".dev")
    if not is_dev and mta_version < (0, 1, 3) or mta_version >= (0, 2, 0):
        warnings.warn(
            f"Found metatomic.torch v{metatomic.torch.__version__}, but "
            "vesin.metatomic was only tested with metatomic.torch >=0.1.3,<0.2. "
            "This may lead to unexpected errors. Please install a compatible version "
            "of metatomic.torch if you encounter any issues.",
            stacklevel=1,
        )


_check_mta_version()

from ._model import (  # noqa: E402
    compute_requested_neighbors,
    compute_requested_neighbors_from_options,
    neighbor_lists_for_model,
)
from ._neighbors import NeighborList  # noqa: E402


__all__ = [
    "NeighborList",
    "compute_requested_neighbors",
    "compute_requested_neighbors_from_options",
    "neighbor_lists_for_model",
]
