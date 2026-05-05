from typing import Any

import torch


class _FakeState:
    """
    Fake version of the ``vesin::State`` TorchBind class, used by ``torch.compile``
    and ``torch.export`` while tracing. The state does not influence the shape of the
    outputs, so this class does not need to contain anything.
    """

    @classmethod
    def __obj_unflatten__(cls, flattened: Any) -> "_FakeState":
        return cls()


def _neighbors_fake(
    state: Any,
    points: torch.Tensor,
    box: torch.Tensor,
    periodic: torch.Tensor,
    cutoff: float,
    full_list: bool,
    sorted: bool,
    algorithm: str,
    skin: float,
    n_threads: int,
    return_shifts: bool,
    return_distances: bool,
    return_vectors: bool,
    copy: bool,
):
    """
    Fake implementation of ``vesin::neighbors``, used by ``torch.compile`` to get the
    shape, dtype and device of the outputs. This must match the C++ implementation.
    """
    n_pairs = torch.library.get_ctx().new_dynamic_size()
    device = points.device

    pairs = torch.empty((n_pairs, 2), dtype=torch.int64, device=device)

    n_shifts = n_pairs if return_shifts else 0
    shifts = torch.empty((n_shifts, 3), dtype=torch.int32, device=device)

    n_distances = n_pairs if return_distances else 0
    distances = torch.empty((n_distances,), dtype=torch.float64, device=device)

    n_vectors = n_pairs if return_vectors else 0
    vectors = torch.empty((n_vectors, 3), dtype=torch.float64, device=device)

    return pairs, shifts, distances, vectors


TORCH_VERSION = tuple(int(v) for v in torch.__version__.split(".")[:2])

# first version of torch where `torch.compile` and `torch.export` can be used with
# vesin_torch. Older versions can still use it in eager mode and with TorchScript.
MINIMAL_TORCH_VERSION_FOR_COMPILE = (2, 5)


def register_compile_support():
    if TORCH_VERSION < MINIMAL_TORCH_VERSION_FOR_COMPILE:
        return

    torch._library.register_fake_class("vesin::State")(_FakeState)
    torch.library.register_fake("vesin::neighbors", _neighbors_fake)

    # Operators taking a TorchBind object are considered to have side effects by
    # default, and wrapped in `with_effects` to keep them ordered. The only side
    # effect of `vesin::neighbors` is to update the cache in the state, and the
    # results do not depend on the order of the calls, so we opt out of this.
    if TORCH_VERSION >= (2, 10):
        torch.library._register_effectful_op("vesin::neighbors", None)
    else:
        from torch._higher_order_ops.effects import _register_effectful_op

        _register_effectful_op(torch.ops.vesin.neighbors.default, None)
