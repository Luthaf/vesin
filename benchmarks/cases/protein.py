from functools import partial
from pathlib import Path

import ase.io
import zstandard

from .base import BenchmarkCase, CaseStructure, PairOptions


_DATA_DIR = Path(__file__).resolve().parent / "data"
_PDB_CODES = ["2JOF", "1LYZ", "3D8E", "3RAW", "7KLL", "1R1R", "1SS8", "1AON"]


def _load(code: str):
    path = _DATA_DIR / f"{code}.xyz.zst"
    with zstandard.open(str(path), "rt") as f:
        atoms = ase.io.read(f, format="extxyz")

    # These structures use PDB coordinates, i.e. multiples of 0.01 A, which creates
    # pairs sitting exactly at the cutoff distance. Whether such a pair is included then
    # depends on floating point rounding inside each implementation, and different
    # implementations disagree on the number of pairs. Adding a tiny amount of noise
    # removes these exact ties, while leaving the structures physically unchanged.
    atoms.rattle(stdev=1e-6, seed=0xDEADBEEF)

    return atoms


class ProteinCase(BenchmarkCase):
    """Benchmark solvated proteins from the PDB, in order of increasing size."""

    name = "protein"

    options = [
        PairOptions(cutoff=6.0, full_list=True),
        PairOptions(cutoff=12.0, full_list=True),
    ]

    def structures(self):
        structures = []
        for pdb_code in _PDB_CODES:
            build = partial(_load, pdb_code)
            structures.append(CaseStructure(label=pdb_code, build=build))

        return structures
