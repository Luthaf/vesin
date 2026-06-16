import argparse
import json
import time

import ase.build
import numpy as np
import torch

import vesin


HAS_CUDA = torch.cuda.is_available()


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


def setup_nnpops_cpu(atoms, cutoff):
    positions = torch.tensor(atoms.positions)
    box_vector = torch.tensor(atoms.cell[:])
    return positions, cutoff, box_vector


def setup_nnpops_cuda(atoms, cutoff):
    positions = torch.tensor(atoms.positions).to("cuda")
    box_vector = torch.tensor(atoms.cell[:]).to("cuda")
    return positions, cutoff, box_vector


def nnpops_run(positions, cutoff, box_vectors):
    import NNPOps.neighbors

    return NNPOps.neighbors.getNeighborPairs(
        positions, cutoff=cutoff, box_vectors=box_vectors
    )


def setup_nvalchemi_cpu(atoms, cutoff):
    positions = torch.tensor(atoms.positions)
    cell = torch.tensor(atoms.cell[:]).unsqueeze(0)
    pbc = torch.tensor(atoms.pbc)
    return positions, cutoff, cell, pbc


def setup_nvalchemi_cuda(atoms, cutoff):
    positions = torch.tensor(atoms.positions, device="cuda")
    cell = torch.tensor(atoms.cell[:], device="cuda").unsqueeze(0)
    pbc = torch.tensor(atoms.pbc, device="cuda")
    return positions, cutoff, cell, pbc


def nvalchemi_run(positions, cutoff, cell, pbc):
    from nvalchemiops.torch.neighbors import neighbor_list as nvalchemi_neighbor_list

    return nvalchemi_neighbor_list(positions, cutoff, cell=cell, pbc=pbc)


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


def ase_run(quantities, atoms, cutoff):
    import ase.neighborlist

    return ase.neighborlist.neighbor_list(quantities, atoms, cutoff)


def matscipy_run(quantities, atoms, cutoff):
    import matscipy.neighbours

    return matscipy.neighbours.neighbour_list(
        cutoff=cutoff,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
        quantities=quantities,
    )


def setup_sisl(atoms, cutoff):
    import sisl

    return sisl.Geometry.new.ase(atoms), cutoff


def sisl_run(system, cutoff):
    import sisl

    finder = sisl.geom.NeighborFinder(system, R=cutoff)
    finder.find_neighbors()


def setup_vesin_cpu(atoms, cutoff):
    calculator = vesin.NeighborList(
        cutoff=cutoff,
        full_list=True,
        sorted=False,
    )
    return calculator, atoms.positions, atoms.cell[:], atoms.pbc


def setup_vesin_cuda(atoms, cutoff):
    calculator = vesin.NeighborList(
        cutoff=cutoff,
        full_list=True,
        sorted=False,
    )
    return (
        calculator,
        torch.tensor(atoms.positions).to("cuda"),
        torch.tensor(atoms.cell[:]).to("cuda"),
        torch.tensor(atoms.pbc).to("cuda"),
    )


def vesin_run(calculator, positions, cell, pbc):
    return calculator.compute(positions, cell, pbc, quantities="ijSd", copy=False)


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


def get_version(impl):
    if impl == "ase":
        import ase

        return ase.__version__
    elif impl == "matscipy":
        import matscipy

        return matscipy.__version__
    elif impl == "pymatgen":
        import pymatgen.core

        return pymatgen.core.__version__
    elif impl == "sisl":
        import sisl

        return sisl.__version__
    elif impl == "vesin_cpu" or impl == "vesin_cuda":
        return vesin.__version__
    elif impl == "nvalchemi_cpu" or impl == "nvalchemi_cuda":
        import nvalchemiops

        return nvalchemiops.__version__
    elif impl == "nnpops_cpu" or impl == "nnpops_cuda":
        import NNPOps

        return NNPOps.__version__
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def run_benchmark(impl, atoms, cutoff):
    if impl == "ase":
        return benchmark(setup_ase_like, ase_run, atoms, cutoff)
    elif impl == "matscipy":
        return benchmark(setup_ase_like, matscipy_run, atoms, cutoff)
    elif impl == "pymatgen":
        return benchmark(setup_pymatgen, pymatgen_run, atoms, cutoff)
    elif impl == "sisl":
        return benchmark(setup_sisl, sisl_run, atoms, cutoff)
    elif impl == "vesin_cpu":
        return benchmark(setup_vesin_cpu, vesin_run, atoms, cutoff)
    elif impl == "vesin_cuda":
        return benchmark(setup_vesin_cuda, vesin_run, atoms, cutoff)
    elif impl == "nvalchemi_cpu":
        return benchmark(setup_nvalchemi_cpu, nvalchemi_run, atoms, cutoff)
    elif impl == "nvalchemi_cuda":
        return benchmark(setup_nvalchemi_cuda, nvalchemi_run, atoms, cutoff)
    elif impl == "nnpops_cpu":
        if np.any(atoms.cell.lengths() < 2 * cutoff):
            print("   NNPOps can not run for this super cell")
            return float("nan")
        else:
            return benchmark(setup_nnpops_cpu, nnpops_run, atoms, cutoff)
    elif impl == "nnpops_cuda":
        if np.any(atoms.cell.lengths() < 2 * cutoff):
            print("   NNPOps can not run for this super cell")
            return float("nan")
        else:
            return benchmark(setup_nnpops_cuda, nnpops_run, atoms, cutoff)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark neighbor list implementations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for benchmark results (default: no output)",
    )
    parser.add_argument(
        "implementations",
        nargs="*",
        default=["all"],
        help="List of implementations to benchmark (default: all)",
    )
    args = parser.parse_args()

    if args.implementations == ["all"]:
        args.implementations = [
            "ase",
            "matscipy",
            "pymatgen",
            "sisl",
            "vesin_cpu",
            "nvalchemi_cpu",
            "nnpops_cpu",
        ]

        if HAS_CUDA:
            args.implementations.append("vesin_cuda")
            args.implementations.append("nvalchemi_cuda")
            args.implementations.append("nnpops_cuda")

    atoms = ase.build.bulk("C", "diamond", 3.567, orthorhombic=True)
    repeats = determine_super_cell(
        atoms, max_cell_repeat=20, max_log_size_delta=0.1, max_cell_ratio=3
    )

    data = {"n_atoms": []}
    for impl in args.implementations:
        data[impl] = {
            "version": get_version(impl),
            "cutoff_3": [],
            "cutoff_6": [],
            "cutoff_12": [],
        }

    structures = []
    for kx, ky, kz in repeats:
        super_cell = atoms.repeat((kx, ky, kz))
        print(f"\n{len(super_cell)} atoms")
        data["n_atoms"].append(len(super_cell))

        for cutoff in [3.0, 6.0, 12.0]:
            print(f"Cutoff: {cutoff} Å")

            for impl in args.implementations:
                timing = run_benchmark(impl, super_cell, cutoff=cutoff)
                data[impl][f"cutoff_{int(cutoff)}"].append(timing)
                print(f"    {impl} took {1e3 * timing:.4f} ms")

    if args.output is not None:
        json.dump(data, open(args.output, "w"), indent=4)
