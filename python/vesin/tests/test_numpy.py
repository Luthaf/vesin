import itertools
import os
import time

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


@pytest.mark.parametrize(
    "system",
    ["water", "diamond", "naphthalene", "carbon", "slab", "Cd2I4O12", "rotated_box"],
)
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


def test_slab_slow():
    # This system was taking >1s due to an issue in the number of cells searched
    atoms = ase.io.read(f"{CURRENT_DIR}/data/slab.xyz")

    start = time.time()
    vesin.ase_neighbor_list("ijSD", atoms, cutoff=5.0)
    end = time.time()
    elsapsed_time = end - start

    # should be closer to 10ms in practice, but allow
    # for some leeway for debug builds & co.
    assert elsapsed_time < 0.01


def test_pairs_output():
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")

    calculator = NeighborList(cutoff=2.0, full_list=True, sorted=False)
    i, j, P = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ijP"
    )

    assert np.all(np.vstack([i, j]).T == P)


def test_array_like_inputs():
    calculator = NeighborList(cutoff=1.0, full_list=True)

    i, j, D, d = calculator.compute(
        points=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        box=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        periodic=False,
        quantities="ijDd",
    )

    assert i.tolist() == [0, 1]
    assert j.tolist() == [1, 0]
    assert np.allclose(d, [0.5, 0.5])
    assert np.allclose(D, [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])


def test_ase_neighbor_list_integer_cutoff():
    atoms = ase.Atoms(
        positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        cell=2.0 * np.eye(3),
        pbc=False,
    )

    i, j = vesin.ase_neighbor_list("ij", atoms, cutoff=1)

    assert i.tolist() == [0, 1]
    assert j.tolist() == [1, 0]


def test_sorting():
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")

    calculator = NeighborList(cutoff=2.0, full_list=True, sorted=False)
    (unsorted_i,) = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="i"
    )
    assert not np.all(np.sort(unsorted_i) == unsorted_i)

    calculator = NeighborList(cutoff=2.0, full_list=True, sorted=True)
    (sorted_i,) = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="i"
    )
    assert np.all(sorted_i == np.sort(sorted_i))

    # check that unsorted is not already sorted by chance
    assert not np.all(sorted_i == unsorted_i)

    # https://github.com/Luthaf/vesin/issues/34
    atoms = ase.io.read(f"{CURRENT_DIR}/data/Cd2I4O12.xyz")
    calculator = NeighborList(cutoff=5.0, full_list=True, sorted=True)
    (sorted_i,) = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="i"
    )
    assert np.all(sorted_i == np.sort(sorted_i))


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


def test_cpu_brute_force_error():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
    box = np.eye(3, dtype=np.float64) * 10.0

    with pytest.raises(
        RuntimeError,
        match="only VesinAutoAlgorithm and VesinCellList are supported on CPU",
    ):
        nl = NeighborList(
            cutoff=1.0,
            full_list=False,
            algorithm="brute_force",
        )
        nl.compute(points, box, periodic=False, quantities="ij")


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_empty_neighbors(dtype):
    """Dtype must be preserved even when no pairs are found (n_pairs == 0)."""
    box = np.eye(3, dtype=dtype) * 10.0
    # Two points far apart with a tiny cutoff: guaranteed 0 neighbors
    points = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=dtype)

    calculator = NeighborList(cutoff=0.1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, False, "ijSDd")

    assert len(i) == 0
    assert D.dtype == dtype
    assert d.dtype == dtype


def test_integer_input_float_outputs():
    box = np.eye(3, dtype=np.int64) * 10
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)

    calculator = NeighborList(cutoff=2, full_list=True)
    i, j, D, d = calculator.compute(points, box, False, "ijDd")

    assert i.dtype == np.uint64
    assert j.dtype == np.uint64
    assert D.dtype == np.float64
    assert d.dtype == np.float64
    assert np.allclose(d, [1.0, 1.0])
    assert np.allclose(D, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def test_integer_input_empty_float_outputs():
    box = np.eye(3, dtype=np.int64) * 10
    points = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.int64)

    calculator = NeighborList(cutoff=1, full_list=True)
    D, d = calculator.compute(points, box, False, "Dd")

    assert D.dtype == np.float64
    assert d.dtype == np.float64


def test_gigantic_cell():
    """Check that the code properly handles very large periodic cells"""
    cell = 1e7 * np.eye(3)

    np.random.seed(34)
    positions = np.random.randn(3000, 3) * 100

    calc = NeighborList(cutoff=8.0, full_list=True)

    (P,) = calc.compute(positions, cell, periodic=True, quantities="P")
    assert len(P) == 412


def test_all_negative_coordinates():
    points = np.array(
        [
            [3.87166034, 1.63896135, 22.14371739],
            [3.19075223, 1.6182371, 22.87858863],
            [5.01141342, 1.73904134, 20.79729257],
        ]
    )
    translated = points - 15.0
    assert np.all(translated[:, 0] < 0)
    assert np.all(translated[:, 1] < 0)

    box = np.zeros((3, 3), dtype=np.float64)

    expected_pairs = {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }

    calculator = NeighborList(cutoff=8.0, full_list=True)

    original_i, original_j = calculator.compute(points, box, False, "ij")
    translated_i, translated_j = calculator.compute(translated, box, False, "ij")

    assert set(zip(original_i, original_j, strict=True)) == (expected_pairs)
    assert set(zip(translated_i, translated_j, strict=True)) == (expected_pairs)
