import ase
import ase.build
import ase.neighborlist
import numpy as np

import vesin


def test_neighbors():
    atoms = ase.build.make_supercell(ase.build.bulk("Si", "diamond"), 3 * np.eye(3))

    ase_nl = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff=3.5)
    ase_nl = [(i, j, S, D) for i, j, S, D in zip(*ase_nl)]

    vesin_nl = vesin.ase_neighbor_list("ijSD", atoms, cutoff=3.5)
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
