from functools import partial

import ase
import numpy as np

from .base import BenchmarkCase, CaseStructure, PairOptions


def _graphite_slabs(nx, ny):
    """Two graphite slabs (3 sheets each) separated by 6 A of vacuum in z.

    Periodic in the x/y plane, non-periodic in z.
    """
    a = 2.46  # in-plane lattice constant (A)
    interlayer = 2.8  # distance between graphite sheets (A)
    vacuum = 6.0  # vacuum gap between the two slabs (A)

    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.5 * a, np.sqrt(3.0) / 2.0 * a, 0.0])

    # two atoms per hexagonal cell, in fractional coordinates
    basis = np.array([[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]])

    positions = []
    for slab_z0 in (0.0, 2 * interlayer + vacuum):
        for sheet in range(3):
            shift = (a1 + a2) / 3.0 if sheet % 2 == 1 else np.zeros(3)
            z = slab_z0 + sheet * interlayer
            for ix in range(nx):
                for iy in range(ny):
                    for fx, fy in basis:
                        pos = fx * a1 + fy * a2 + ix * a1 + iy * a2 + shift
                        positions.append([pos[0], pos[1], z])

    positions = np.asarray(positions)

    height = 4 * interlayer + vacuum
    cell = np.array(
        [
            nx * a1,
            ny * a2,
            [0.0, 0.0, height],
        ]
    )
    atoms = ase.Atoms(positions=positions, cell=cell, pbc=[True, True, False])
    atoms.rattle(stdev=0.1, seed=0xDEADBEEF)
    return atoms


class GraphiteCase(BenchmarkCase):
    """Benchmark two graphite slabs (3 sheets each) with 6 A vacuum, x/y periodic."""

    name = "graphite"

    options = [
        PairOptions(cutoff=4.5, full_list=True),
        PairOptions(cutoff=9.0, full_list=True),
    ]

    def structures(self):
        # equally spaced number of in-plane cells in log scale
        repeats = np.logspace(np.log10(2), np.log10(100), num=10)

        structures = []
        for repeat in repeats:
            nx = ny = int(round(repeat))
            build = partial(_graphite_slabs, nx, ny)
            structures.append(CaseStructure(label=str(12 * nx * ny), build=build))

        return structures
