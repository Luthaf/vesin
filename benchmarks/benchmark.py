import json
import sys
import time

import ase.build
import ase.neighborlist
import numpy as np


def benchmark(setup, function, atoms, cutoff):
    try:
        args = setup(atoms, cutoff)
    except Exception as e:
        print("failed to run setup:", file=sys.stderr)
        print(e, file=sys.stderr)
        return float("nan")

    n_warm = 3
    start = time.time()
    try:
        for _ in range(n_warm):
            function(*args)
    except Exception as e:
        print("failed to run function:", file=sys.stderr)
        print(e, file=sys.stderr)
        return float("nan")
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

    try:
        for _ in range(n_iter):
            function(*args)
    except Exception as e:
        print("failed to run function:", file=sys.stderr)
        print(e, file=sys.stderr)
        return float("nan")

    end = time.time()

    return (end - start) / n_iter


def setup_vesin_cpu(atoms, cutoff):
    points = np.asarray(atoms.positions)
    box = np.asarray(atoms.cell[:])
    periodic = np.asarray(atoms.pbc)

    return cutoff, points, box, periodic, "ijSd"


def setup_vesin_cuda(atoms, cutoff):
    import cupy as cp

    points = cp.asarray(atoms.positions)
    box = cp.asarray(atoms.cell[:])
    periodic = cp.asarray(atoms.pbc)

    return cutoff, points, box, periodic, "ijSd"


def vesin_run(cutoff, points, box, periodic, quantities):
    import vesin

    nl = vesin.NeighborList(cutoff, full_list=True, algorithm="auto")

    return nl.compute(
        points=points,
        box=box,
        periodic=periodic,
        quantities=quantities,
        copy=False,
    )


def matscipy_run(quantities, atoms, cutoff):
    import matscipy.neighbours

    return matscipy.neighbours.neighbour_list(quantities, atoms, cutoff)


def setup_torch_nl_cpu(atoms, cutoff):
    import torch
    import torch_nl

    pos, cell, pbc, batch, _ = torch_nl.ase2data([atoms], device=torch.device("cpu"))
    return cutoff, pos, cell, pbc, batch


def setup_torch_nl_cuda(atoms, cutoff):
    import torch
    import torch_nl

    pos, cell, pbc, batch, _ = torch_nl.ase2data([atoms], device=torch.device("cuda"))
    return cutoff, pos, cell, pbc, batch


def torch_nl_run(cutoff, pos, cell, pbc, batch):
    import torch_nl

    return torch_nl.compute_neighborlist(
        cutoff, pos, cell, pbc, batch, self_interaction=True
    )


def setup_nnpops_cpu(atoms, cutoff):
    import torch

    positions = torch.tensor(atoms.positions)
    box_vector = torch.tensor(atoms.cell[:])
    return positions, cutoff, box_vector


def setup_nnpops_cuda(atoms, cutoff):
    import torch

    positions = torch.tensor(atoms.positions).to("cuda")
    box_vector = torch.tensor(atoms.cell[:]).to("cuda")
    return positions, cutoff, box_vector


def nnpops_run(positions, cutoff, box_vectors):
    import NNPOps.neighbors

    return NNPOps.neighbors.getNeighborPairs(
        positions, cutoff=cutoff, box_vectors=box_vectors
    )


def setup_ase_like(atoms, cutoff):
    return "ijSd", atoms, float(cutoff)


def setup_pymatgen(atoms, cutoff):
    import pymatgen.core

    structure = pymatgen.core.Structure(
        atoms.cell[:],
        atoms.numbers,
        atoms.positions,
        coords_are_cartesian=True,
    )
    return structure, cutoff


def pymatgen_run(structure, cutoff):
    return structure.get_neighbor_list(cutoff)


def setup_sisl(atoms, cutoff):
    import sisl

    return sisl.Geometry.new.ase(atoms), cutoff


def sisl_run(system, cutoff):
    import sisl

    finder = sisl.geom.NeighborFinder(system, R=cutoff)
    finder.find_neighbors()


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
sisl_time = {}
torch_nl_cpu_time = {}
torch_nl_cuda_time = {}
nnpops_cpu_time = {}
nnpops_cuda_time = {}
pymatgen_time = {}
vesin_cpu_time = {}
vesin_cuda_time = {}

if len(sys.argv) > 1:
    CUTOFFS = list(map(int, sys.argv[1:]))
else:
    CUTOFFS = [3, 6, 12]

for cutoff in CUTOFFS:
    print(f"===========  CUTOFF={cutoff}  =============")

    n_atoms[cutoff] = []
    ase_time[cutoff] = []
    matscipy_time[cutoff] = []
    sisl_time[cutoff] = []
    torch_nl_cpu_time[cutoff] = []
    torch_nl_cuda_time[cutoff] = []
    nnpops_cpu_time[cutoff] = []
    nnpops_cuda_time[cutoff] = []
    pymatgen_time[cutoff] = []
    vesin_cpu_time[cutoff] = []
    vesin_cuda_time[cutoff] = []

    for kx, ky, kz in repeats:
        super_cell = atoms.repeat((kx, ky, kz))
        super_cell.positions[:] += 0.2 * np.random.random(super_cell.positions.shape)
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
            matscipy_run,
            super_cell,
            cutoff,
        )
        matscipy_time[cutoff].append(timing * 1e3)
        print(f"   matscipy took {timing * 1e3:.3f} ms")

        # SISL
        timing = benchmark(
            setup_sisl,
            sisl_run,
            super_cell,
            cutoff,
        )
        sisl_time[cutoff].append(timing * 1e3)
        print(f"   sisl took {timing * 1e3:.3f} ms")

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
            nnpops_cpu_time[cutoff].append(float("nan"))
            nnpops_cuda_time[cutoff].append(float("nan"))
        else:
            # NNPOps CPU
            timing = benchmark(
                setup_nnpops_cpu,
                nnpops_run,
                super_cell,
                cutoff,
            )
            nnpops_cpu_time[cutoff].append(timing * 1e3)
            print(f"   NNPOps (cpu) took {timing * 1e3:.3f} ms")

            # NNPOps CUDA
            timing = benchmark(
                setup_nnpops_cuda,
                nnpops_run,
                super_cell,
                cutoff,
            )
            nnpops_cuda_time[cutoff].append(timing * 1e3)
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

        # VESIN CPU
        timing = benchmark(
            setup_vesin_cpu,
            vesin_run,
            super_cell,
            cutoff,
        )
        vesin_cpu_time[cutoff].append(timing * 1e3)
        print(f"   vesin (cpu) took {timing * 1e3:.3f} ms")

        # VESIN CUDA
        timing = benchmark(
            setup_vesin_cuda,
            vesin_run,
            super_cell,
            cutoff,
        )
        vesin_cuda_time[cutoff].append(timing * 1e3)
        print(f"   vesin (cuda) took {timing * 1e3:.3f} ms")
        print()


for cutoff in CUTOFFS:
    data = {
        "n_atoms": n_atoms[cutoff],
        "ase": ase_time[cutoff],
        "matscipy": matscipy_time[cutoff],
        "torch_nl_cpu": torch_nl_cpu_time[cutoff],
        "torch_nl_cuda": torch_nl_cuda_time[cutoff],
        "nnpops_cpu": nnpops_cpu_time[cutoff],
        "nnpops_cuda": nnpops_cuda_time[cutoff],
        "pymatgen": pymatgen_time[cutoff],
        "sisl": sisl_time[cutoff],
        "vesin_cpu": vesin_cpu_time[cutoff],
        "vesin_cuda": vesin_cuda_time[cutoff],
    }
    with open(f"cutoff-{cutoff}.json", "w") as fd:
        json.dump(data, fd)
