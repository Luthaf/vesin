import warnings


try:
    from vesin_torch import NeighborList, __version__
except ImportError as e:
    raise ImportError(
        "vesin_torch is not installed. Please install it with `pip install vesin-torch`"
    ) from e


def __getattr__(name: str):
    warnings.warn(
        (
            "`vesin.torch` is deprecated and will be removed in a future release. "
            "Please import code from `vesin_torch` instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    if name == "__version__":
        return __version__
    elif name == "NeighborList":
        return NeighborList

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
