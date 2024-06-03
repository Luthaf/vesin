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

    ase_nl = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)
    ase_nl = [(i, j, S, D) for i, j, S, D in zip(*ase_nl)]

    vesin_nl = vesin.ase_neighbor_list("ijSD", atoms, cutoff)
    vesin_nl = [(i, j, S, D) for i, j, S, D in zip(*vesin_nl)]

    assert len(ase_nl) == len(vesin_nl)

    for i, j, S, D in vesin_nl:
        found = False
        for ref_i, ref_j, ref_S, ref_D in ase_nl:
            if i == ref_i and j == ref_j and np.all(S == ref_S):
                assert np.allclose(D, ref_D)
                found = True
                break

        assert found
