from functools import partial

import ase
import numpy as np

from .base import BenchmarkCase, CaseStructure, PairOptions


def fcc_droplet(n_cells, lattice):
    """A roughly spherical FCC cluster, centered at the origin."""
    basis = np.asarray([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    r = []
    span = range(-n_cells, n_cells + 1)

    for ix in span:
        for iy in span:
            for iz in span:
                for b in basis:
                    r.append((np.asarray([ix, iy, iz]) + b) * lattice)

    r = np.asarray(r, dtype=float)
    r -= r.mean(axis=0)
    radius = n_cells * lattice
    return r[(r * r).sum(axis=1) <= radius * radius]


def _droplet_atoms(n_cells):
    """Create an ASE atoms object for an FCC droplet of the given size."""
    lattice = 2.1
    positions = fcc_droplet(n_cells, lattice)

    box = 2 * (n_cells * lattice + 12.0)
    atoms = ase.Atoms(positions=positions, cell=[box, box, box], pbc=False)
    atoms.rattle(stdev=0.1, seed=0xDEADBEEF)
    return atoms


class NanodropletCase(BenchmarkCase):
    """Benchmark spherical FCC nanodroplets of increasing sizes."""

    name = "nanodroplet"

    options = [
        PairOptions(cutoff=2.5, full_list=True),
        PairOptions(cutoff=5.0, full_list=True),
    ]

    def structures(self):
        # scores evenly spaced in log space from 10 up to 1M atoms
        targets = np.logspace(np.log10(10), np.log10(1_000_000), num=10)

        structures = []
        for target in targets:
            # atom count of the FCC sphere scales as 16/3 * pi * n_cells**3
            n_cells = max(1, int(round((target / (16.0 * np.pi / 3.0)) ** (1.0 / 3.0))))
            build = partial(_droplet_atoms, n_cells)
            structures.append(CaseStructure(label=str(int(target)), build=build))

        return structures
