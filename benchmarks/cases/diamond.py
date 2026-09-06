from functools import partial

import ase.build
import numpy as np

from .base import BenchmarkCase, CaseStructure, PairOptions


def determine_super_cell(atoms, max_cell_repeat, max_log_size_delta, max_cell_ratio):
    """
    Determine which super cells to include. We want equally spaced number of atoms in
    log scale, and cells that are not too anisotropic.
    """
    sizes = {}
    for kx in range(1, max_cell_repeat):
        for ky in range(1, max_cell_repeat):
            for kz in range(1, max_cell_repeat):
                size = kx * ky * kz
                if size in sizes:
                    sizes[size].append((kx, ky, kz))
                else:
                    sizes[size] = [(kx, ky, kz)]

    # for each size, pick the less anisotropic cell
    repeats = []
    a, b, c = atoms.cell.lengths()
    for candidates in sizes.values():
        best = None
        best_ratio = np.inf
        for kx, ky, kz in candidates:
            lengths = [kx * a, ky * b, kz * c]
            ratio = np.max(lengths) / np.min(lengths)
            if ratio < best_ratio:
                best = (kx, ky, kz)
                best_ratio = ratio

        repeats.append(best)

    filtered_repeats = []
    filtered_log_sizes = [-1]

    for kx, ky, kz in repeats:
        log_size = np.log(kx * ky * kz)
        lengths = [kx * a, ky * b, kz * c]
        ratio = np.max(lengths) / np.min(lengths)
        if np.min(np.abs(np.array(filtered_log_sizes) - log_size)) > max_log_size_delta:
            if log_size < 2 or ratio < max_cell_ratio:
                filtered_repeats.append((kx, ky, kz))
                filtered_log_sizes.append(log_size)

    return filtered_repeats


def _build_diamond(sizes):
    atoms = ase.build.bulk("C", "diamond", 3.567, orthorhombic=True)
    atoms = atoms.repeat(sizes)
    atoms.rattle(stdev=0.1, seed=0xDEADBEEF)
    return atoms


class DiamondCase(BenchmarkCase):
    """Benchmark increasing sizes of diamond supercells."""

    name = "diamond"

    options = [
        PairOptions(cutoff=3.0, full_list=True),
        PairOptions(cutoff=6.0, full_list=True),
        PairOptions(cutoff=12.0, full_list=True),
        PairOptions(cutoff=3.0, full_list=False),
        PairOptions(cutoff=6.0, full_list=False),
        PairOptions(cutoff=12.0, full_list=False),
    ]

    def structures(self):
        atoms = ase.build.bulk("C", "diamond", 3.567, orthorhombic=True)
        repeats = determine_super_cell(
            atoms,
            max_cell_repeat=40,
            max_log_size_delta=0.5,
            max_cell_ratio=3,
        )

        structures = []
        for kx, ky, kz in repeats:
            build = partial(_build_diamond, (kx, ky, kz))
            structures.append(
                CaseStructure(label=str(len(atoms) * kx * ky * kz), build=build)
            )

        return structures
