import os

import ase.build
import ase.io
import ase.neighborlist
import numpy as np
import pytest

import vesin

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("system", ["water", "diamond", "naphthalene", "carbon"])
@pytest.mark.parametrize("quantities", ["ijSD", "ijS"])
@pytest.mark.parametrize("cutoff", [float(i) for i in range(1, 10)])
def test_neighbors(system, quantities, cutoff):
    atoms = ase.io.read(f"{CURRENT_DIR}/data/{system}.xyz")

    ase_nl = ase.neighborlist.neighbor_list(quantities, atoms, cutoff)
    if quantities == "ijS":
        ase_nl = [(i, j, S) for i, j, S in zip(*ase_nl)]
    elif quantities == "ijSD":
        ase_nl = [(i, j, S, D) for i, j, S, D in zip(*ase_nl)]
    else:
        raise ValueError(f"Unknown quantities: {quantities}")

    vesin_nl = vesin.ase_neighbor_list(quantities, atoms, cutoff)
    if quantities == "ijS":
        vesin_nl = [(i, j, S) for i, j, S in zip(*vesin_nl)]
    elif quantities == "ijSD":
        vesin_nl = [(i, j, S, D) for i, j, S, D in zip(*vesin_nl)]

    assert len(ase_nl) == len(vesin_nl)

    for sample in vesin_nl:
        if quantities == "ijS":
            i, j, S = sample
        elif quantities == "ijSD":
            i, j, S, D = sample
        else:
            raise ValueError(f"Unknown quantities: {quantities}")
        found = False
        for ref_sample in ase_nl:
            if quantities == "ijS":
                ref_i, ref_j, ref_S = ref_sample
            elif quantities == "ijSD":
                ref_i, ref_j, ref_S, ref_D = ref_sample
            else:
                raise ValueError(f"Unknown quantities: {quantities}")
            if i == ref_i and j == ref_j and np.all(S == ref_S):
                found = True
                if "D" in quantities:
                    assert np.allclose(D, ref_D)
                break

        assert found
