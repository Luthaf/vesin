import itertools
import os

import ase.io
import ase.neighborlist
import numpy as np
import pytest

import vesin
from vesin import NeighborList


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def non_sorted_nl(quantities, atoms, cutoff):
    calculator = NeighborList(cutoff=cutoff, full_list=True, sorted=False)
    outputs = calculator.compute(
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=atoms.pbc,
        quantities=quantities,
        copy=False,
    )
    # since we have `copy=False`, also return the calculator to keep the memory alive
    return *outputs, calculator


@pytest.mark.parametrize("system", ["water", "diamond", "naphthalene", "carbon"])
@pytest.mark.parametrize("cutoff", [float(i) for i in range(1, 10)])
@pytest.mark.parametrize("vesin_nl", [vesin.ase_neighbor_list, non_sorted_nl])
def test_neighbors(system, cutoff, vesin_nl):
    atoms = ase.io.read(f"{CURRENT_DIR}/data/{system}.xyz")

    ase_i, ase_j, ase_S, ase_D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)
    vesin_i, vesin_j, vesin_S, vesin_D, *_ = vesin_nl("ijSD", atoms, cutoff)

    assert len(ase_i) == len(vesin_i)
    assert len(ase_j) == len(vesin_j)
    assert len(ase_S) == len(vesin_S)
    assert len(ase_D) == len(vesin_D)

    ase_ijS = np.concatenate(
        (ase_i.reshape(-1, 1), ase_j.reshape(-1, 1), ase_S), axis=1
    )
    vesin_ijS = np.concatenate(
        (vesin_i.reshape(-1, 1), vesin_j.reshape(-1, 1), vesin_S), axis=1
    )

    ase_sort_indices = np.lexsort(ase_ijS.T)
    vesin_sort_indices = np.lexsort(vesin_ijS.T)

    assert np.array_equal(ase_ijS[ase_sort_indices], vesin_ijS[vesin_sort_indices])
    assert np.allclose(ase_D[ase_sort_indices], vesin_D[vesin_sort_indices])


def test_pairs_output():
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")

    calculator = NeighborList(cutoff=2.0, full_list=True, sorted=False)
    i, j, P = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ijP"
    )

    assert np.all(np.vstack([i, j]).T == P)


def test_sorting():
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")

    calculator = NeighborList(cutoff=2.0, full_list=True, sorted=False)
    i, j = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ij"
    )
    unsorted_ij = np.concatenate((i.reshape(-1, 1), j.reshape(-1, 1)), axis=1)
    assert not np.all(unsorted_ij[np.lexsort((j, i))] == unsorted_ij)

    calculator = NeighborList(cutoff=2.0, full_list=True, sorted=True)
    i, j = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ij"
    )

    sorted_ij = np.concatenate((i.reshape(-1, 1), j.reshape(-1, 1)), axis=1)
    assert np.all(sorted_ij[np.lexsort((j, i))] == sorted_ij)

    # check that unsorted is not already sorted by chance
    assert not np.all(sorted_ij == unsorted_ij)

    # https://github.com/Luthaf/vesin/issues/34
    atoms = ase.io.read(f"{CURRENT_DIR}/data/Cd2I4O12.xyz")
    calculator = NeighborList(cutoff=5.0, full_list=True, sorted=True)
    i, j = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ij"
    )
    sorted_ij = np.concatenate((i.reshape(-1, 1), j.reshape(-1, 1)), axis=1)
    assert np.all(sorted_ij[np.lexsort((j, i))] == sorted_ij)


def test_errors():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    box = np.zeros((3, 3))

    nl = NeighborList(cutoff=1.2, full_list=True)

    message = "the box matrix is not invertible"
    with pytest.raises(RuntimeError, match=message):
        nl.compute(points, box, periodic=True, quantities="ij")

    box = np.eye(3, 3)
    message = "cutoff is too small"
    with pytest.raises(RuntimeError, match=message):
        nl = NeighborList(cutoff=1e-12, full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")

    message = "cutoff must be a finite, positive number"
    with pytest.raises(RuntimeError, match=message):
        nl = NeighborList(cutoff=0.0, full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")

    with pytest.raises(RuntimeError, match=message):
        nl = NeighborList(cutoff=-12.0, full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")

    with pytest.raises(RuntimeError, match=message):
        nl = NeighborList(cutoff=float("inf"), full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")

    with pytest.raises(RuntimeError, match=message):
        nl = NeighborList(cutoff=float("nan"), full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")


@pytest.mark.parametrize(
    "periodic",
    list(itertools.product([False, True], repeat=3)),
)
def test_mixed_periodic(periodic):
    cutoff = 0.35
    # the box is not a cube to better test the periodic conditions
    box = np.eye(3, dtype=np.float64)[[2, 0, 1]] + 0.1 * np.random.normal(size=(3, 3))
    points = np.random.default_rng(0).random((100, 3))

    atoms = ase.Atoms(positions=points, cell=box, pbc=periodic)
    ase_i, ase_j, ase_S, ase_D, ase_d = ase.neighborlist.neighbor_list(
        "ijSDd", atoms, cutoff
    )

    calculator = NeighborList(cutoff=cutoff, full_list=True)
    vesin_i, vesin_j, vesin_S, vesin_D, vesin_d = calculator.compute(
        points, box, periodic, "ijSDd"
    )

    assert len(ase_i) == len(vesin_i)
    assert len(ase_j) == len(vesin_j)
    assert len(ase_S) == len(vesin_S)
    assert len(ase_D) == len(vesin_D)
    assert len(ase_d) == len(vesin_d)

    ase_ijS = np.concatenate(
        (ase_i.reshape(-1, 1), ase_j.reshape(-1, 1), ase_S), axis=1
    )
    vesin_ijS = np.concatenate(
        (vesin_i.reshape(-1, 1), vesin_j.reshape(-1, 1), vesin_S), axis=1
    )

    ase_sort_indices = np.lexsort(np.flip(ase_ijS, axis=1).T)
    vesin_sort_indices = np.lexsort(np.flip(vesin_ijS, axis=1).T)

    assert np.array_equal(ase_ijS[ase_sort_indices], vesin_ijS[vesin_sort_indices])
    assert np.allclose(ase_D[ase_sort_indices], vesin_D[vesin_sort_indices])
    assert np.allclose(ase_d[ase_sort_indices], vesin_d[vesin_sort_indices])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype(dtype):
    box = np.eye(3, dtype=dtype) * 3.0
    points = np.random.default_rng(0).random((100, 3), dtype=dtype) * 3.0

    calculator = NeighborList(cutoff=4, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, True, "ijSDd")

    assert i.dtype == np.uint64
    assert j.dtype == np.uint64
    assert s.dtype == np.int32
    assert D.dtype == dtype
    assert d.dtype == dtype
