import itertools
import os

import ase.io
import ase.neighborlist
import numpy as np
import pytest

from vesin import NeighborList


torch = pytest.importorskip("torch")


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("full_list", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_large_box_small_cutoff(full_list, device):
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-6.0, 0.0, 0.0],
            [-6.0, -2.0, 0.0],
            [-6.0, 0.0, -2.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    box = torch.tensor(
        [
            [54.0, 0.0, 0.0],
            [0.0, 54.0, 0.0],
            [0.0, 0.0, 54.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    calculator = NeighborList(cutoff=2.1, full_list=full_list)

    i, j, S, d, D = calculator.compute(points, box, periodic=True, quantities="ijSdD")

    i = i.to(dtype=torch.int64)
    j = j.to(dtype=torch.int64)

    # check all outputs are torch tensors
    assert isinstance(i, torch.Tensor)
    assert isinstance(j, torch.Tensor)
    assert isinstance(S, torch.Tensor)
    assert isinstance(d, torch.Tensor)
    assert isinstance(D, torch.Tensor)

    assert len(i) == len(j)
    assert len(i) == len(d)
    assert len(i) == len(D)
    assert len(i) == len(S)
    assert len(i) == (8 if full_list else 4)

    pairs = torch.stack((i, j), axis=1)
    sort_idx = torch.argsort(pairs[:, 0] * (i.max() + 1) + pairs[:, 1])

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

        expected_shifts = torch.zeros(
            (8, 3),
            dtype=torch.int32,
            device=device,
        )
        expected_distances = torch.tensor(
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            dtype=torch.float64,
            device=device,
        )
        expected_vectors = torch.tensor(
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
            dtype=torch.float64,
            device=device,
        )
    else:
        expected_pairs = sorted([(0, 1), (0, 2), (3, 4), (3, 5)])
        expected_shifts = torch.zeros(
            (4, 3),
            dtype=torch.int32,
            device=device,
        )
        expected_distances = torch.tensor(
            [2.0, 2.0, 2.0, 2.0],
            dtype=torch.float64,
            device=device,
        )
        expected_vectors = torch.tensor(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -2.0],
            ],
            dtype=torch.float64,
            device=device,
        )

    assert actual_pairs == expected_pairs
    assert torch.all(S == expected_shifts)
    assert torch.allclose(d, expected_distances)
    assert torch.allclose(D, expected_vectors)


@pytest.mark.parametrize("system", ["water", "diamond", "naphthalene", "carbon"])
@pytest.mark.parametrize("algorithm", ["brute_force", "cell_list", "auto"])
@pytest.mark.parametrize("device", DEVICES)
def test_neighbors(system, algorithm, device):
    if algorithm == "brute_force" and device == "cpu":
        # not implemented
        return

    atoms = ase.io.read(f"{CURRENT_DIR}/../../vesin/tests/data/{system}.xyz")

    # make the cell bigger for MIC
    if not np.allclose(atoms.cell, np.zeros((3, 3))):
        atoms = atoms.repeat((2, 2, 2))

    cutoff = 2.0

    ase_i, ase_j, ase_S, ase_D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)

    calculator = NeighborList(cutoff=cutoff, full_list=True, algorithm=algorithm)
    vesin_i, vesin_j, vesin_S, vesin_D = calculator.compute(
        points=torch.tensor(
            atoms.positions,
            device=device,
        ),
        box=torch.tensor(
            atoms.cell[:],
            device=device,
        ),
        periodic=torch.tensor(
            atoms.pbc,
            device=device,
        ),
        quantities="ijSD",
    )

    # get as numpy arrays
    vesin_i = vesin_i.cpu().numpy()
    vesin_j = vesin_j.cpu().numpy()
    vesin_S = vesin_S.cpu().numpy()
    vesin_D = vesin_D.cpu().numpy()

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


@pytest.mark.parametrize(
    "periodic",
    list(itertools.product([False, True], repeat=3)),
)
@pytest.mark.parametrize("algorithm", ["brute_force", "cell_list", "auto"])
@pytest.mark.parametrize("device", DEVICES)
def test_mixed_periodic(periodic, algorithm, device):
    if algorithm == "brute_force" and device == "cpu":
        # not implemented
        return

    cutoff = 0.35
    # Use a fixed seed for the box to ensure reproducibility
    rng = np.random.default_rng(42)
    box = np.eye(3, dtype=np.float64)[[2, 0, 1]] + 0.1 * rng.normal(size=(3, 3))
    points = np.random.default_rng(0).random((100, 3))

    atoms = ase.Atoms(positions=points, cell=box, pbc=periodic)
    ase_i, ase_j, ase_S, ase_D, ase_d = ase.neighborlist.neighbor_list(
        "ijSDd", atoms, cutoff
    )

    calculator = NeighborList(cutoff=cutoff, full_list=True, algorithm=algorithm)
    vesin_i, vesin_j, vesin_S, vesin_D, vesin_d = calculator.compute(
        points=torch.tensor(points, dtype=torch.float64, device=device),
        box=torch.tensor(box, dtype=torch.float64, device=device),
        periodic=torch.tensor(periodic, device=device),
        quantities="ijSDd",
    )

    vesin_i = vesin_i.cpu().numpy()
    vesin_j = vesin_j.cpu().numpy()
    vesin_S = vesin_S.cpu().numpy()
    vesin_D = vesin_D.cpu().numpy()
    vesin_d = vesin_d.cpu().numpy()

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


@pytest.mark.parametrize("device", DEVICES)
def test_no_neighbors(device):
    """Test implementation when there are no neighbors"""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64, device=device
    )
    box = torch.eye(3, dtype=torch.float64, device=device)

    calculator = NeighborList(cutoff=0.1, full_list=True)
    i, j = calculator.compute(points, box, True, quantities="ij")

    assert len(i) == 0
    assert len(j) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_dtype(dtype, device):
    box = torch.eye(3, dtype=dtype, device=device) * 3.0
    points = torch.tensor(
        np.random.default_rng(0).random((100, 3)) * 3.0,
        dtype=dtype,
        device=device,
    )

    calculator = NeighborList(cutoff=1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, True, "ijSDd")

    assert i.dtype == torch.uint64
    assert j.dtype == torch.uint64
    assert s.dtype == torch.int32
    assert D.dtype == dtype
    assert d.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_dtype_empty_neighbors(dtype, device):
    """Dtype must be preserved even when no pairs are found (n_pairs == 0)."""
    box = torch.eye(3, dtype=dtype, device=device) * 10.0
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=dtype, device=device
    )

    calculator = NeighborList(cutoff=0.1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, False, "ijSDd")

    assert len(i) == 0
    assert D.dtype == dtype
    assert d.dtype == dtype
