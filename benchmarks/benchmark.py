import time

import ase.build
import ase.neighborlist
import matscipy.neighbours
import NNPOps.neighbors
import numpy as np
import pymatgen.core
import torch
import torch_nl

import vesin


def benchmark(setup, function, atoms, cutoff):
    args = setup(atoms, cutoff)

    n_warm = 5
    start = time.time()
    for _ in range(n_warm):
        function(*args)
    end = time.time()

    warmup = (end - start) / n_warm

    # dynamically pick the number of iterations to keep timing below 1s per test, while
    # also ensuring at least 10 repetitions
    n_iter = int(1.0 / warmup)
    if n_iter > 10000:
        n_iter = 10000
    elif n_iter < 10:
        n_iter = 10

    start = time.time()
    for _ in range(n_iter):
        function(*args)
    end = time.time()

    return (end - start) / n_iter


def setup_torch_nl_cpu(atoms, cutoff):
    pos, cell, pbc, batch, n_atoms = torch_nl.ase2data(
        [atoms], device=torch.device("cpu")
    )
    return cutoff, pos, cell, pbc, batch


def setup_torch_nl_cuda(atoms, cutoff):
    pos, cell, pbc, batch, n_atoms = torch_nl.ase2data(
        [atoms], device=torch.device("cuda")
    )
    return cutoff, pos, cell, pbc, batch


def torch_nl_run(cutoff, pos, cell, pbc, batch):
    return torch_nl.compute_neighborlist(
        cutoff, pos, cell, pbc, batch, self_interaction=True
    )


def setup_nnpops_cpu(atoms, cutoff):
    positions = torch.tensor(atoms.positions)
    box_vector = torch.tensor(atoms.cell)
    return positions, cutoff, box_vector


def setup_nnpops_cuda(atoms, cutoff):
    positions = torch.tensor(atoms.positions).to("cuda")
    box_vector = torch.tensor(atoms.cell).to("cuda")
    return positions, cutoff, box_vector


def nnpops_run(positions, cutoff, box_vectors):
    return NNPOps.neighbors.getNeighborPairs(
        positions, cutoff=cutoff, box_vectors=box_vectors
    )


def setup_ase_like(atoms, cutoff):
    return "ijSd", atoms, float(cutoff)


def setup_pymatgen(atoms, cutoff):
    structure = pymatgen.core.Structure(
        atoms.cell[:],
        atoms.numbers,
        atoms.positions,
        coords_are_cartesian=True,
    )
    return structure, cutoff


def pymatgen_run(structure, cutoff):
    return structure.get_neighbor_list(cutoff)


def determine_super_cell(max_cell_repeat, max_log_size_delta, max_cell_ratio):
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


atoms = ase.build.bulk("C", "diamond", 3.567, orthorhombic=True)

repeats = determine_super_cell(
    max_cell_repeat=20, max_log_size_delta=0.1, max_cell_ratio=3
)


n_atoms = {}
ase_time = {}
matscipy_time = {}
torch_nl_cpu_time = {}
torch_nl_cuda_time = {}
pymatgen_time = {}
vesin_time = {}

for cutoff in [3, 6, 12]:
    print(f"===========  CUTOFF={cutoff}  =============")

    n_atoms[cutoff] = []
    ase_time[cutoff] = []
    matscipy_time[cutoff] = []
    torch_nl_cpu_time[cutoff] = []
    torch_nl_cuda_time[cutoff] = []
    pymatgen_time[cutoff] = []
    vesin_time[cutoff] = []

    for kx, ky, kz in repeats:
        super_cell = atoms.repeat((kx, ky, kz))
        print(len(super_cell), "atoms")
        n_atoms[cutoff].append(len(super_cell))

        # ASE
        timing = benchmark(
            setup_ase_like,
            ase.neighborlist.neighbor_list,
            super_cell,
            cutoff,
        )
        ase_time[cutoff].append(timing * 1e3)
        print(f"   ase took {timing * 1e3:.3f} ms")

        # MATSCIPY
        timing = benchmark(
            setup_ase_like,
            matscipy.neighbours.neighbour_list,
            super_cell,
            cutoff,
        )
        matscipy_time[cutoff].append(timing * 1e3)
        print(f"   matscipy took {timing * 1e3:.3f} ms")

        # TORCH_NL CPU
        timing = benchmark(
            setup_torch_nl_cpu,
            torch_nl_run,
            super_cell,
            cutoff,
        )
        torch_nl_cpu_time[cutoff].append(timing * 1e3)
        print(f"   torch_nl (cpu) took {timing * 1e3:.3f} ms")

        # TORCH_NL CUDA
        timing = benchmark(
            setup_torch_nl_cuda,
            torch_nl_run,
            super_cell,
            cutoff,
        )
        torch_nl_cuda_time[cutoff].append(timing * 1e3)
        print(f"   torch_nl (cuda) took {timing * 1e3:.3f} ms")

        if np.any(super_cell.cell.lengths() < 2 * cutoff):
            print("   NNPOps can not run for this super cell")
        else:
            # NNPOps CPU
            timing = benchmark(
                setup_nnpops_cpu,
                nnpops_run,
                super_cell,
                cutoff,
            )
            torch_nl_cpu_time[cutoff].append(timing * 1e3)
            print(f"   NNPOps (cpu) took {timing * 1e3:.3f} ms")

            # NNPOps CUDA
            timing = benchmark(
                setup_nnpops_cuda,
                nnpops_run,
                super_cell,
                cutoff,
            )
            torch_nl_cuda_time[cutoff].append(timing * 1e3)
            print(f"   NNPOps (cuda) took {timing * 1e3:.3f} ms")

        # Pymatgen
        timing = benchmark(
            setup_pymatgen,
            pymatgen_run,
            super_cell,
            cutoff,
        )
        pymatgen_time[cutoff].append(timing * 1e3)
        print(f"   pymatgen took {timing * 1e3:.3f} ms")

        # VESIN
        timing = benchmark(
            setup_ase_like,
            vesin.ase_neighbor_list,
            super_cell,
            cutoff,
        )
        vesin_time[cutoff].append(timing * 1e3)
        print(f"   vesin took {timing * 1e3:.3f} ms")
        print()
