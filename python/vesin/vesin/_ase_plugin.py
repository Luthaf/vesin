"""ASE neighbour-list plugin for Vesin (host + experimental device capability).

Registers Vesin as an ``ase.plugins`` neighbour-list backend so it can be
selected via :func:`ase.neighborlist.get_neighbor_list` (ASE v4 plugin API).
Selection is never automatic -- this only makes the backend *available* under the
name ``"vesin"``.

This module is the ``ase.plugins`` entry point (it exposes ``__ase_plugins__``).
Registration is guarded so installing Vesin alongside an ASE that predates the v4
plugin API registers nothing rather than breaking plugin discovery. The heavy
imports (ase, CuPy) happen lazily, inside the adapters.
"""

from __future__ import annotations


def neighbor_list(quantities, atoms, cutoff, *, self_interaction=False):
    """Vesin neighbour list adapted to ASE's ``NeighborListFunction`` protocol.

    Thin wrapper over :func:`vesin.ase_neighbor_list`, returning the same
    ``(i, j, d, D, S)`` flat arrays and quantity letters as
    :func:`ase.neighborlist.neighbor_list`. Vesin has no ``self_interaction``
    option (it never returns pure self-pairs); per the plugin contract a request
    it cannot honour is rejected rather than silently differing.
    """
    if self_interaction:
        raise NotImplementedError(
            "the vesin backend does not support self_interaction=True"
        )
    from ._ase import ase_neighbor_list

    return ase_neighbor_list(quantities, atoms, cutoff)


def device_neighbor_list(device_id=0):
    """Return the *experimental* device-resident Vesin backend.

    The result satisfies ASE's experimental ``DeviceNeighborList`` protocol
    (:mod:`ase._4.plugins.neighborlist_device`) and builds edge data on-device
    (exchanged via DLPack). GPU-only (CUDA) and requires CuPy.

    The device capability is separate from the host ``neighbor_list``
    registration: the host ``NeighborListPlugin`` carries no device slot, so a
    consumer obtains the device backend through this factory and checks
    ``isinstance(backend, DeviceNeighborList)``.
    """
    from ._ase_device import VesinDeviceNeighborList

    return VesinDeviceNeighborList(device_id=device_id)


try:
    from ase._4.plugins.neighborlist import NeighborListPlugin
except ImportError:
    # ASE without the v4 plugin API: register nothing, but do not break the
    # discovery of other plugins.
    __ase_plugins__: set = set()
else:
    __ase_plugins__ = {
        NeighborListPlugin(
            "vesin",
            long_name="Vesin neighbour list",
            citation="https://github.com/Luthaf/vesin",
            implementation="vesin._ase_plugin.neighbor_list",
        ),
    }
