from functools import partial

import ase.build
import numpy as np

from .base import BenchmarkCase, CaseStructure, PairOptions
from .diamond import determine_super_cell


def _slanted_unit_cell():
    atoms = ase.build.bulk("C", "diamond", 3.567, orthorhombic=True)
    slant = np.array([[1, 0, 0], [3, 1, 0], [3, 3, 1]])
    return ase.build.make_supercell(atoms, slant)


def _build_triclinic(sizes):
    atoms = _slanted_unit_cell()
    atoms = atoms.repeat(sizes)
    atoms.rattle(stdev=0.1, seed=0xDEADBEEF)
    return atoms


class TriclinicCase(BenchmarkCase):
    """Benchmark increasing sizes of diamond supercells in a very slanted cell."""

    name = "triclinic"

    options = [
        PairOptions(cutoff=3.0, full_list=True),
        PairOptions(cutoff=6.0, full_list=True),
        PairOptions(cutoff=12.0, full_list=True),
        PairOptions(cutoff=3.0, full_list=False),
        PairOptions(cutoff=6.0, full_list=False),
        PairOptions(cutoff=12.0, full_list=False),
    ]

    def structures(self):
        atoms = _slanted_unit_cell()
        repeats = determine_super_cell(
            atoms,
            max_cell_repeat=40,
            max_log_size_delta=0.5,
            max_cell_ratio=3,
        )

        structures = []
        for kx, ky, kz in repeats:
            build = partial(_build_triclinic, (kx, ky, kz))
            structures.append(
                CaseStructure(label=str(len(atoms) * kx * ky * kz), build=build)
            )

        return structures
