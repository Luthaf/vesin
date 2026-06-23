import ase.build
import ase.neighborlist
import numpy as np
import pytest

from vesin._ase_plugin import device_neighbor_list, neighbor_list


def _ijS_set(i, j, S):
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    S = np.asarray(S, dtype=np.int64)
    return set(
        zip(
            i.tolist(),
            j.tolist(),
            *[S[:, k].tolist() for k in range(3)],
            strict=True,
        )
    )


def _system():
    atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    return atoms


# --------------------------------------------------------------------------- #
# Host plugin (no GPU)
# --------------------------------------------------------------------------- #
def test_host_matches_ase():
    atoms = _system()
    cutoff = 4.0
    vi, vj, vd, vD, vS = neighbor_list("ijdDS", atoms, cutoff)
    ai, aj, ad, aD, aS = ase.neighborlist.neighbor_list("ijdDS", atoms, cutoff)
    assert len(vi) == len(ai)
    assert _ijS_set(vi, vj, vS) == _ijS_set(ai, aj, aS)


def test_host_self_interaction_rejected():
    atoms = _system()
    with pytest.raises(NotImplementedError):
        neighbor_list("ij", atoms, 4.0, self_interaction=True)


def test_plugin_registration():
    # Exposes __ase_plugins__; the "vesin" plugin is present iff ASE has the v4
    # plugin API (older ASE -> empty set, by design).
    from vesin import _ase_plugin

    try:
        from ase._4.plugins.neighborlist import NeighborListPlugin  # noqa: F401
    except ImportError:
        assert _ase_plugin.__ase_plugins__ == set()
    else:
        names = {p.name for p in _ase_plugin.__ase_plugins__}
        assert "vesin" in names


# --------------------------------------------------------------------------- #
# Device backend (CUDA via CuPy + ASE v4 device protocol)
# --------------------------------------------------------------------------- #
def _device_skip_reason():
    try:
        import cupy as cp
    except ImportError as exc:
        return f"cupy not available: {exc}"
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover - env dependent
        return f"CUDA not available: {exc}"
    try:
        import ase._4.plugins.neighborlist_device  # noqa: F401
    except ImportError:
        return "ASE has no experimental device neighbour-list protocol"
    return None


_DEVICE_SKIP = _device_skip_reason()
device = pytest.mark.skipif(_DEVICE_SKIP is not None, reason=str(_DEVICE_SKIP))


@device
def test_device_satisfies_protocol():
    from ase._4.plugins.neighborlist_device import DeviceNeighborList

    be = device_neighbor_list()
    assert isinstance(be, DeviceNeighborList)
    assert be.differentiable is False
    assert be.device[0] == 2  # CUDA


@device
def test_device_equivalence_vs_ase():
    import cupy as cp

    atoms = _system()
    cutoff = 4.0
    res = device_neighbor_list().build_device(
        cp.asarray(atoms.positions),
        np.asarray(atoms.cell[:]),
        tuple(bool(b) for b in atoms.pbc),
        cutoff,
        "ijSD",
    )
    vi = cp.asnumpy(res.get("i"))
    vj = cp.asnumpy(res.get("j"))
    vS = cp.asnumpy(res.get("S"))
    assert res.get("i").__dlpack_device__()[0] == 2  # device-resident

    ai, aj, aS, aD = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)
    assert _ijS_set(vi, vj, vS) == _ijS_set(ai, aj, aS)


@device
def test_device_needs_rebuild_delegates():
    # Reuse is delegated to Vesin's persistent NeighborList(skin=); needs_rebuild
    # is a no-op device-resident True (the consumer always calls build_device).
    import cupy as cp

    atoms = _system()
    be = device_neighbor_list()
    be.build_device(
        cp.asarray(atoms.positions),
        np.asarray(atoms.cell[:]),
        tuple(bool(b) for b in atoms.pbc),
        4.0,
        "ijS",
    )
    nr = be.needs_rebuild(cp.asarray(atoms.positions), skin=0.4)
    assert nr.__dlpack_device__()[0] == 2  # device-resident
    assert bool(nr) is True


@device
def test_device_skin_reuse_correct():
    # With skin>0 the persistent calculator reuses the list across a sub-skin
    # move; the result must still match a fresh build (Vesin filters to the true
    # cutoff on reuse).
    import cupy as cp

    from vesin._ase_device import VesinDeviceNeighborList

    atoms = _system()
    cell = np.asarray(atoms.cell[:])
    pbc = tuple(bool(b) for b in atoms.pbc)
    be = VesinDeviceNeighborList(skin=0.5)
    be.build_device(cp.asarray(atoms.positions), cell, pbc, 4.0, "ijS")  # seed cache
    moved = atoms.positions + 0.05  # sub-skin displacement -> reuse path
    res = be.build_device(cp.asarray(moved), cell, pbc, 4.0, "ijS")
    vi = cp.asnumpy(res.get("i"))
    vj = cp.asnumpy(res.get("j"))
    vS = cp.asnumpy(res.get("S"))

    a2 = atoms.copy()
    a2.positions = moved
    ai, aj, aS = ase.neighborlist.neighbor_list("ijS", a2, 4.0)
    assert _ijS_set(vi, vj, vS) == _ijS_set(ai, aj, aS)


@device
def test_device_padded_unsupported():
    import cupy as cp

    atoms = _system()
    be = device_neighbor_list()
    with pytest.raises(NotImplementedError):
        be.build_device(
            cp.asarray(atoms.positions),
            np.asarray(atoms.cell[:]),
            tuple(bool(b) for b in atoms.pbc),
            4.0,
            max_capacity=64,
        )
