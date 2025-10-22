import itertools
import os

import ase.io
import ase.neighborlist
import numpy as np
import pytest

from vesin import NeighborList


cp = pytest.importorskip("cupy")

# Check if CUDA is available
try:
    cp.cuda.Device(0).compute_capability
    # all good
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA is not available", allow_module_level=True)


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("full_list", [False, True])
def test_large_box_small_cutoff(full_list):
    points = cp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-6.0, 0.0, 0.0],
            [-6.0, -2.0, 0.0],
            [-6.0, 0.0, -2.0],
        ],
        dtype=cp.float64,
    )

    box = cp.array(
        [
            [54.0, 0.0, 0.0],
            [0.0, 54.0, 0.0],
            [0.0, 0.0, 54.0],
        ],
        dtype=cp.float64,
    )

    calculator = NeighborList(cutoff=2.1, full_list=full_list)

    i, j, S, d, D = calculator.compute(points, box, periodic=True, quantities="ijSdD")

    # check all outputs are cupy arrays
    assert isinstance(i, cp.ndarray)
    assert isinstance(j, cp.ndarray)
    assert isinstance(S, cp.ndarray)
    assert isinstance(d, cp.ndarray)
    assert isinstance(D, cp.ndarray)

    assert len(i) == len(j)
    assert len(i) == len(d)
    assert len(i) == len(D)
    assert len(i) == len(S)
    assert len(i) == (8 if full_list else 4)

    pairs = cp.stack((i, j), axis=1)
    sort_idx = cp.argsort(pairs[:, 0] * (i.max() + 1) + pairs[:, 1])

    # Apply sort
    i = i[sort_idx]
    j = j[sort_idx]
    S = S[sort_idx]
    d = d[sort_idx]
    D = D[sort_idx]

    # Convert to plain Python lists for easy matching
    actual_pairs = sorted(zip(i.tolist(), j.tolist(), strict=True))

    if full_list:
        expected_pairs = [
            (0, 1),
            (0, 2),
            (1, 0),
            (2, 0),
            (3, 4),
            (3, 5),
            (4, 3),
            (5, 3),
        ]

        expected_shifts = cp.zeros((8, 3), dtype=cp.int32)
        expected_distances = cp.array(
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            dtype=cp.float64,
        )
        expected_vectors = cp.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -2.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -2.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=cp.float64,
        )
    else:
        expected_pairs = sorted([(0, 1), (0, 2), (3, 4), (3, 5)])
        expected_shifts = cp.zeros((4, 3), dtype=cp.int32)
        expected_distances = cp.array([2.0, 2.0, 2.0, 2.0], dtype=cp.float64)
        expected_vectors = cp.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -2.0],
            ],
            dtype=cp.float64,
        )

    assert actual_pairs == expected_pairs
    assert cp.all(S == expected_shifts)
    assert cp.allclose(d, expected_distances)
    assert cp.allclose(D, expected_vectors)


# FIXME: re-enable 'diamond' and 'carbon' tests for CUDA
# @pytest.mark.parametrize("system", ["water", "diamond", "naphthalene", "carbon"])
@pytest.mark.parametrize("system", ["water", "naphthalene"])
def test_neighbors(system):
    atoms = ase.io.read(f"{CURRENT_DIR}/../../vesin/tests/data/{system}.xyz")

    # make the cell bigger for MIC
    if not np.allclose(atoms.cell, np.zeros((3, 3))):
        atoms = atoms.repeat((2, 2, 2))

    cutoff = 2.0

    ase_i, ase_j, ase_S, ase_D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)

    calculator = NeighborList(cutoff=cutoff, full_list=True)
    vesin_i, vesin_j, vesin_S, vesin_D = calculator.compute(
        points=cp.array(atoms.positions),
        box=cp.array(atoms.cell[:]),
        periodic=cp.array(atoms.pbc),
        quantities="ijSD",
    )

    # get as numpy arrays
    vesin_i = cp.asnumpy(vesin_i)
    vesin_j = cp.asnumpy(vesin_j)
    vesin_S = cp.asnumpy(vesin_S)
    vesin_D = cp.asnumpy(vesin_D)

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

    ase_sort_indices = np.lexsort(np.flip(ase_ijS, axis=1).T)
    vesin_sort_indices = np.lexsort(np.flip(vesin_ijS, axis=1).T)

    assert np.array_equal(ase_ijS[ase_sort_indices], vesin_ijS[vesin_sort_indices])
    assert np.allclose(ase_D[ase_sort_indices], vesin_D[vesin_sort_indices])


@pytest.mark.skip(reason="cuda implementataion is currently broken")
@pytest.mark.parametrize(
    "periodic",
    list(itertools.product([False, True], repeat=3)),
)
def test_mixed_periodic(periodic):
    cutoff = 0.35
    box = np.eye(3, dtype=np.float64)
    points = np.random.default_rng(0).random((100, 3))

    atoms = ase.Atoms(positions=points, cell=box, pbc=periodic)
    ase_i, ase_j, ase_S, ase_D, ase_d = ase.neighborlist.neighbor_list(
        "ijSDd", atoms, cutoff
    )

    calculator = NeighborList(cutoff=cutoff, full_list=True)
    vesin_i, vesin_j, vesin_S, vesin_D, vesin_d = calculator.compute(
        points=cp.array(points, dtype=cp.float64),
        box=cp.array(box, dtype=cp.float64),
        periodic=cp.array(periodic),
        quantities="ijSDd",
    )

    vesin_i = cp.asnumpy(vesin_i)
    vesin_j = cp.asnumpy(vesin_j)
    vesin_S = cp.asnumpy(vesin_S)
    vesin_D = cp.asnumpy(vesin_D)
    vesin_d = cp.asnumpy(vesin_d)

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


def test_no_neighbors():
    """Test CUDA implementation when there are no neighbors"""
    points = cp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=cp.float64)
    box = cp.eye(3, dtype=cp.float64)

    calculator_gpu = NeighborList(cutoff=0.1, full_list=True)
    i, j = calculator_gpu.compute(points, box, True, quantities="ij")

    assert len(i) == 0
    assert len(j) == 0
