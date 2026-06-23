"""Experimental: device-resident neighbour-list adapter for ASE.

Adapts Vesin's CUDA cell list (reached through its CuPy array interface) to ASE's
experimental device neighbour-list protocol
(:mod:`ase._4.plugins.neighborlist_device`): :class:`DeviceNeighborList` and
:class:`DeviceNeighborResult`. A device-resident calculator discovers it via
``isinstance(backend, DeviceNeighborList)`` and exchanges edge data on-device via
DLPack, with no host round-trip per step.

GPU-only and requires CuPy: passing CuPy device positions to
:meth:`vesin.NeighborList.compute` dispatches to the CUDA kernel and returns
device-resident CuPy arrays. The host path is unchanged (``vesin.ase_neighbor_list``).

Notes / limitations
-------------------
* ``differentiable = False`` here only because this adapter goes through the
  **CuPy** path, which is not an autograd framework -- not a Vesin limitation.
  Vesin's torch binding (``vesin-torch``) runs on GPU and is autograd-
  differentiable, so it would make a natural ``differentiable = True`` device
  backend for this protocol (a useful follow-up).
* No fixed-capacity / dense output -- Vesin currently returns COO only, so the
  padded (``max_capacity``) path is unsupported and raises. (Dense output is in
  progress upstream; the padded path can map onto it once available.)
* ``needs_rebuild`` returns an on-device boolean scalar (a CuPy max-squared-
  displacement reduction; no host sync inside the call) so a compiled consumer
  can branch on rebuild-vs-reuse without leaving the device. Vesin's
  ``NeighborList`` already rebuilds on device as needed via its ``skin=`` Verlet
  reuse, but does not expose that decision as a queryable scalar; if it did, this
  adapter would use it directly instead of the explicit reduction.
* Scalar cutoff only; ``self_interaction=True`` rejected -- both raise.
"""

import numpy as np


def _cupy():
    import cupy as cp  # imported lazily; this adapter is GPU-only

    return cp


def _to_device_cell(cell):
    """Return the (3, 3) cell as a CuPy array (Vesin needs box and points to be
    the same array type). The cell is tiny, so any host->device copy here is not
    the residency concern (that is the O(n_atoms) positions / O(n_edges) edges)."""
    cp = _cupy()
    if isinstance(cell, cp.ndarray):
        return cell
    to_dlpack_device = getattr(cell, "__dlpack_device__", None)
    if to_dlpack_device is not None and to_dlpack_device()[0] != 1:  # device array
        return cp.from_dlpack(cell)
    return cp.asarray(np.ascontiguousarray(np.asarray(cell), dtype=float))


class VesinDeviceResult:
    """On-device neighbour data (CuPy); satisfies ``DeviceNeighborResult``.

    Vesin returns COO arrays only, so this result is never padded.
    """

    def __init__(self, arrays, *, n_edges):
        self._arrays = arrays  # name -> CuPy device array (DLPack-exporting)
        self._n_edges = n_edges

    @property
    def n_edges(self):
        return int(self._n_edges)

    @property
    def did_overflow(self):
        return False

    @property
    def padded(self):
        return False

    def get(self, quantity):
        try:
            return self._arrays[quantity]
        except KeyError:
            available = ", ".join(sorted(self._arrays)) or "(none)"
            raise KeyError(
                f"quantity {quantity!r} not available; have {available}. "
                "It was not among the requested quantities."
            ) from None

    def mask(self):
        return None


class VesinDeviceNeighborList:
    """Vesin device backend; satisfies ``DeviceNeighborList``."""

    differentiable = False

    def __init__(self, device_id=0):
        self._device_type = 2  # kDLCUDA (Vesin's GPU backend is CUDA)
        self._device_id = int(device_id)
        self._ref = None  # cached build-time positions (device)

    @property
    def device(self):
        return (self._device_type, self._device_id)

    def build_device(
        self,
        positions,
        cell,
        pbc,
        cutoff,
        quantities="ijS",
        *,
        self_interaction=False,
        max_capacity=None,
        stream=None,
    ):
        if self_interaction:
            raise NotImplementedError(
                "the Vesin backend does not support self_interaction=True"
            )
        if isinstance(cutoff, dict) or np.ndim(cutoff) != 0:
            raise NotImplementedError(
                "the Vesin device backend supports only a scalar cutoff"
            )
        if max_capacity is not None:
            raise NotImplementedError(
                "Vesin has no fixed-capacity (dense) output; use the tight path "
                "(max_capacity=None) for COO i/j/S/D"
            )
        invalid = set(quantities) - set("ijdDS")
        if invalid or not quantities:
            raise ValueError(
                f'quantities must be a non-empty subset of "ijdDS"; '
                f"got {quantities!r}."
            )

        from ._neighbors import NeighborList

        cp = _cupy()
        pos = cp.from_dlpack(positions)  # device array (zero-copy view)
        self._device_id = int(pos.device.id)
        # Snapshot build-time positions on device for needs_rebuild (the consumer
        # may overwrite its positions buffer in place each step).
        self._ref = pos.copy()
        box = _to_device_cell(cell)
        pbc_t = tuple(bool(b) for b in pbc)

        calculator = NeighborList(cutoff=float(cutoff), full_list=True)
        out = calculator.compute(pos, box, pbc_t, quantities=quantities)
        if not isinstance(out, (list, tuple)):  # single-quantity -> bare array
            out = (out,)
        # Keyed by NAME (Vesin returns in requested order; we never rely on it).
        arrays = {name: arr for name, arr in zip(quantities, out)}
        n_edges = int(arrays[quantities[0]].shape[0])
        return VesinDeviceResult(arrays, n_edges=n_edges)

    def needs_rebuild(self, positions, *, skin, stream=None):
        """On-device 0-d bool: max squared displacement > skin**2.

        CuPy reduction (device-resident); the result stays on device -- no host
        sync inside the call. The eager ``bool()`` (CuPy's native ``__bool__``)
        is the single documented sync point; a compiled consumer adopts the
        scalar via ``from_dlpack`` and branches in-graph.
        """
        if self._ref is None:
            raise RuntimeError("call build_device(...) before needs_rebuild(...)")
        cp = _cupy()
        cur = cp.from_dlpack(positions)
        return cp.max(((cur - self._ref) ** 2).sum(axis=1)) > float(skin) ** 2
