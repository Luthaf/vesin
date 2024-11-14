import os

import ase.build
import ase.io
import ase.neighborlist
import numpy as np
import pytest

import vesin


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def non_sorted_nl(quantities, atoms, cutoff):
    calculator = vesin.NeighborList(cutoff=cutoff, full_list=True, sorted=False)
    outputs = calculator.compute(
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=np.all(atoms.pbc),
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


def test_sorting():
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")

    calculator = vesin.NeighborList(cutoff=2.0, full_list=True, sorted=False)
    i, j = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ij"
    )
    unsorted_ij = np.concatenate((i.reshape(-1, 1), j.reshape(-1, 1)), axis=1)
    assert not np.all(unsorted_ij[np.lexsort((j, i))] == unsorted_ij)

    calculator = vesin.NeighborList(cutoff=2.0, full_list=True, sorted=True)
    i, j = calculator.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=True, quantities="ij"
    )

    sorted_ij = np.concatenate((i.reshape(-1, 1), j.reshape(-1, 1)), axis=1)
    assert np.all(sorted_ij[np.lexsort((j, i))] == sorted_ij)

    # check that unsorted is not already sorted by chance
    assert not np.all(sorted_ij == unsorted_ij)


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
