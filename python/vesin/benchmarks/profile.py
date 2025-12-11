#!/usr/bin/env python
"""Profile vesin CUDA kernels with nsys/ncu.

Usage:
    nsys profile python profile.py --algorithm cell_list
    ncu python profile.py --algorithm cell_list --n-runs 1
"""

import numpy as np


try:
    import cupy as cp
except ImportError:
    print("CuPy not available")
    exit(1)

from vesin import NeighborList


def generate_system(n_atoms, density, seed=42):
    """Generate random atomic positions in a cubic box."""
    box_size = (n_atoms / density) ** (1 / 3)
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * box_size
    box = np.eye(3) * box_size
    return positions, box


def profile(algorithm, n_atoms, density, cutoff, n_warmup=5, n_runs=20):
    """Run neighbor list computation for profiling."""
    positions, box = generate_system(n_atoms, density)
    positions_gpu = cp.asarray(positions)
    box_gpu = cp.asarray(box)

    nl = NeighborList(cutoff=cutoff, full_list=True, algorithm=algorithm)

    # Warmup (compile kernels, allocate buffers)
    print(f"Warming up ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
    cp.cuda.Stream.null.synchronize()

    # Profile runs
    print(f"Running {n_runs} iterations for profiling...")
    for _ in range(n_runs):
        i, j = nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
    cp.cuda.Stream.null.synchronize()

    print(f"Found {len(i)} pairs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile vesin CUDA kernels")
    parser.add_argument(
        "--algorithm", choices=["brute_force", "cell_list"], default="cell_list"
    )
    parser.add_argument("--n-atoms", type=int, default=50000)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=20)
    args = parser.parse_args()

    print(f"Algorithm: {args.algorithm}")
    print(f"System: {args.n_atoms} atoms, density={args.density}, cutoff={args.cutoff}")
    print()

    profile(
        args.algorithm,
        args.n_atoms,
        args.density,
        args.cutoff,
        args.n_warmup,
        args.n_runs,
    )

    print()
    print("Done. Use 'nsys profile python profile.py' to generate a timeline.")
