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
* Reuse is delegated to Vesin: the adapter holds a **persistent**
  ``NeighborList(skin=...)`` and ``build_device`` calls ``.compute`` on it, so
  Vesin's own Verlet logic reuses the list across steps (displacement, and box
  changes once Luthaf/vesin#172 lands -- NPT). ``needs_rebuild`` therefore just
  returns a device-resident ``True`` (a no-op signal): the consumer always calls
  ``build_device`` and Vesin makes the reuse cheap internally. The Verlet skin is
  set at construction (``VesinDeviceNeighborList(skin=...)``), matching Vesin's
  API; ``needs_rebuild``'s ``skin`` argument is unused. (A compiled consumer that
  wants to *skip* the build itself would instead need an explicit device-scalar
  displacement check -- an open ASE-protocol design point.)
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

    def __init__(self, device_id=0, skin=0.0):
        self._device_type = 2  # kDLCUDA (Vesin's GPU backend is CUDA)
        self._device_id = int(device_id)
        # Persistent calculator so Vesin's own Verlet ``skin`` reuse kicks in
        # across steps. skin=0 means rebuild every call (no reuse).
        self._skin = float(skin)
        self._nl = None
        self._nl_cutoff = None

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
                f'quantities must be a non-empty subset of "ijdDS"; got {quantities!r}.'
            )

        from ._neighbors import NeighborList

        cp = _cupy()
        pos = cp.from_dlpack(positions)  # device array (zero-copy view)
        self._device_id = int(pos.device.id)
        box = _to_device_cell(cell)
        pbc_t = tuple(bool(b) for b in pbc)

        # Reuse one persistent NeighborList so Vesin's Verlet ``skin`` reuse spans
        # calls (rebuilt only if the cutoff changes). Vesin decides internally
        # whether to rebuild or reuse on each ``.compute``.
        if self._nl is None or self._nl_cutoff != float(cutoff):
            self._nl = NeighborList(
                cutoff=float(cutoff), full_list=True, skin=self._skin
            )
            self._nl_cutoff = float(cutoff)
        out = self._nl.compute(pos, box, pbc_t, quantities=quantities)
        if not isinstance(out, (list, tuple)):  # single-quantity -> bare array
            out = (out,)
        # Keyed by NAME (Vesin returns in requested order; we never rely on it).
        arrays = {name: arr for name, arr in zip(quantities, out, strict=True)}
        n_edges = int(arrays[quantities[0]].shape[0])
        return VesinDeviceResult(arrays, n_edges=n_edges)

    def needs_rebuild(self, positions, *, skin, stream=None):
        """Delegate reuse to Vesin: return a device-resident ``True`` scalar.

        Vesin's persistent ``NeighborList(skin=...)`` owns the rebuild-vs-reuse
        decision internally (displacement, and box changes once Luthaf/vesin#172
        lands), so the adapter always signals "build" and lets ``build_device``'s
        persistent calculator reuse the list cheaply. The ``skin`` argument is not
        used here -- the Verlet skin is set at construction
        (``VesinDeviceNeighborList(skin=...)``), matching Vesin's API. (A compiled
        consumer that needs to *skip* the build itself would instead want an
        explicit device-scalar displacement check; see the ASE protocol notes.)
        """
        cp = _cupy()
        return cp.asarray(True)
