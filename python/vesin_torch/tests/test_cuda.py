import itertools
import os

import ase.io
import ase.neighborlist
import numpy as np
import pytest
import torch

from vesin.torch import NeighborList


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


DEVICES = ["cpu"]
if torch.cuda.is_available():
    for device_id in range(torch.cuda.device_count()):
        DEVICES.append(f"cuda:{device_id}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("full_list", [False, True])
def test_large_box_small_cutoff(device, full_list):
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
    assert len(i) == len(j)
    assert len(i) == len(d)
    assert len(i) == len(D)
    assert len(i) == len(S)
    assert len(i) == (8 if full_list else 4)

    pairs = torch.stack((i, j), dim=1)
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

        expected_shifts = torch.zeros((8, 3), dtype=torch.int32)
        expected_distances = torch.tensor(
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            dtype=torch.float64,
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
        )
    else:
        expected_pairs = sorted([(0, 1), (0, 2), (3, 4), (3, 5)])
        expected_shifts = torch.zeros((4, 3), dtype=torch.int32)
        expected_distances = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
        expected_vectors = torch.tensor(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -2.0],
            ],
            dtype=torch.float64,
        )

    assert actual_pairs == expected_pairs
    assert torch.all(S.cpu() == expected_shifts)
    assert torch.allclose(d.cpu(), expected_distances)
    assert torch.allclose(D.cpu(), expected_vectors)


# FIXME: re-enable 'diamond' and 'carbon' tests for CUDA
# @pytest.mark.parametrize("system", ["water", "diamond", "naphthalene", "carbon"])
@pytest.mark.parametrize("system", ["water", "naphthalene"])
@pytest.mark.parametrize("device", DEVICES)
def test_neighbors(system, device):
    atoms = ase.io.read(f"{CURRENT_DIR}/../../vesin/tests/data/{system}.xyz")

    # make the cell bigger for MIC
    if not np.allclose(atoms.cell, np.zeros((3, 3))):
        atoms = atoms.repeat((2, 2, 2))

    cutoff = 2.0

    ase_i, ase_j, ase_S, ase_D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)

    calculator = NeighborList(cutoff=cutoff, full_list=True)
    vesin_i, vesin_j, vesin_S, vesin_D = calculator.compute(
        points=torch.tensor(atoms.positions).to(device),
        box=torch.tensor(atoms.cell[:]).to(device),
        periodic=torch.tensor(atoms.pbc),
        quantities="ijSD",
    )

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


@pytest.mark.skip(reason="cuda implementataion is currently broken")
@pytest.mark.parametrize(
    "periodic",
    list(itertools.product([False, True], repeat=3)),
)
@pytest.mark.parametrize("device", DEVICES)
def test_mixed_periodic(periodic, device):
    cutoff = 0.35
    box = np.eye(3, dtype=np.float64)
    points = np.random.default_rng(0).random((100, 3))

    atoms = ase.Atoms(positions=points, cell=box, pbc=periodic)
    ase_i, ase_j, ase_S, ase_D, ase_d = ase.neighborlist.neighbor_list(
        "ijSDd", atoms, cutoff
    )

    calculator = NeighborList(cutoff=cutoff, full_list=True)
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
