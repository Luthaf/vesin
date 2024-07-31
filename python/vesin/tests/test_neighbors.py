import os

import ase.build
import ase.io
import ase.neighborlist
import numpy as np
import pytest

import vesin

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("system", ["water", "diamond", "naphthalene", "carbon"])
@pytest.mark.parametrize("cutoff", [float(i) for i in range(1, 10)])
def test_neighbors(system, cutoff):
    atoms = ase.io.read(f"{CURRENT_DIR}/data/{system}.xyz")

    ase_i, ase_j, ase_S, ase_D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)
    vesin_i, vesin_j, vesin_S, vesin_D = vesin.ase_neighbor_list("ijSD", atoms, cutoff)

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


def test_errors():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    box = np.zeros((3, 3))

    nl = vesin.NeighborList(cutoff=1.2, full_list=True)

    message = "the box matrix is not invertible"
    with pytest.raises(RuntimeError, match=message):
        nl.compute(points, box, periodic=True, quantities="ij")

    box = np.eye(3, 3)
    message = "cutoff is too small"
    with pytest.raises(RuntimeError, match=message):
        nl = vesin.NeighborList(cutoff=0.0, full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")

    with pytest.raises(RuntimeError, match=message):
        nl = vesin.NeighborList(cutoff=1e-12, full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")

    with pytest.raises(RuntimeError, match=message):
        nl = vesin.NeighborList(cutoff=-12.0, full_list=True)
        nl.compute(points, box, periodic=True, quantities="ij")
