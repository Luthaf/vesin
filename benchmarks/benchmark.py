import time

import ase.build
import ase.neighborlist
import matscipy.neighbours
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


atoms = ase.build.bulk("C", "diamond", 3.567, orthorhombic=True)

n_atoms = {}
ase_time = {}
matscipy_time = {}
torch_nl_cpu_time = {}
torch_nl_cuda_time = {}
pymatgen_time = {}
vesin_time = {}

for cutoff in [3, 7]:
    print(f"===========  CUTOFF={cutoff}  =============")
    max_cell_repeat = 5

    repeats = []
    seen_sizes = set()
    for kx in range(1, max_cell_repeat):
        for ky in range(1, max_cell_repeat):
            for kz in range(1, max_cell_repeat):
                size = kx * ky * kz
                if size in seen_sizes:
                    continue

                repeats.append((kx, ky, kz))
                seen_sizes.add(kx * ky * kz)

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
