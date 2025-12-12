#!/usr/bin/env python
"""Benchmark vesin neighbor list implementations."""

import time

import numpy as np


try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from vesin import NeighborList


def generate_system(n_atoms, density, seed=42):
    """Generate random atomic positions in a cubic box."""
    box_size = (n_atoms / density) ** (1 / 3)
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * box_size
    box = np.eye(3) * box_size
    return positions, box


def benchmark_cpu(positions, box, cutoff, n_warmup=3, n_runs=5):
    """Benchmark CPU implementation."""
    nl = NeighborList(cutoff=cutoff, full_list=True)

    for _ in range(n_warmup):
        nl.compute(positions, box, periodic=True, quantities="ij")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        i, j = nl.compute(positions, box, periodic=True, quantities="ij")
        times.append(time.perf_counter() - start)

    return np.mean(times) * 1000, np.std(times) * 1000, len(i)


def benchmark_gpu(positions_gpu, box_gpu, cutoff, algorithm, n_warmup=3, n_runs=5):
    """Benchmark GPU implementation using CUDA events for accurate timing."""
    nl = NeighborList(cutoff=cutoff, full_list=True, algorithm=algorithm)

    for _ in range(n_warmup):
        nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
    cp.cuda.Stream.null.synchronize()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    times = []
    for _ in range(n_runs):
        start_event.record()
        i, j = nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
        end_event.record()
        end_event.synchronize()
        times.append(cp.cuda.get_elapsed_time(start_event, end_event))

    return np.mean(times), np.std(times), len(i)


def run_single(n_atoms=50000, density=0.05, cutoff=5.0):
    """Run a single benchmark comparison."""
    positions, box = generate_system(n_atoms, density)

    print(f"System: {n_atoms} atoms, density={density}, cutoff={cutoff}")
    print()

    # CPU
    cpu_time, cpu_std, n_pairs = benchmark_cpu(positions, box, cutoff)
    rate = n_pairs / cpu_time / 1000
    print(f"CPU:        {cpu_time:>8.2f} +/- {cpu_std:.2f} ms  ({rate:.1f} M pairs/ms)")

    if not HAS_CUPY:
        print("CuPy not available, skipping GPU benchmarks")
        return

    positions_gpu = cp.asarray(positions)
    box_gpu = cp.asarray(box)

    # GPU brute force
    bf_time, bf_std, _ = benchmark_gpu(positions_gpu, box_gpu, cutoff, "brute_force")
    rate = n_pairs / bf_time / 1000
    print(f"GPU (BF):   {bf_time:>8.2f} +/- {bf_std:.2f} ms  ({rate:.1f} M pairs/ms)")

    # GPU cell list
    cl_time, cl_std, _ = benchmark_gpu(positions_gpu, box_gpu, cutoff, "cell_list")
    rate = n_pairs / cl_time / 1000
    print(f"GPU (CL):   {cl_time:>8.2f} +/- {cl_std:.2f} ms  ({rate:.1f} M pairs/ms)")

    print()
    print(f"Speedup (CPU/CL): {cpu_time / cl_time:.1f}x")
    print(f"Speedup (BF/CL):  {bf_time / cl_time:.1f}x")
    print(f"Pairs: {n_pairs}")


def run_scaling(density=0.05, cutoff=5.0):
    """Benchmark across different system sizes."""
    if not HAS_CUPY:
        print("CuPy not available")
        return

    print("=" * 80)
    print(f"Scaling Benchmark (density={density}, cutoff={cutoff})")
    print("=" * 80)
    print()
    header = f"{'N atoms':>10} {'N pairs':>12} {'CPU (ms)':>12} "
    header += f"{'GPU BF (ms)':>12} {'GPU CL (ms)':>12} {'Speedup':>10}"
    print(header)
    print("-" * 72)

    for n_atoms in [1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        positions, box = generate_system(n_atoms, density)
        positions_gpu = cp.asarray(positions)
        box_gpu = cp.asarray(box)

        cpu_time, _, n_pairs = benchmark_cpu(positions, box, cutoff)
        bf_time, _, _ = benchmark_gpu(positions_gpu, box_gpu, cutoff, "brute_force")
        cl_time, _, _ = benchmark_gpu(positions_gpu, box_gpu, cutoff, "cell_list")

        speedup = cpu_time / cl_time
        row = f"{n_atoms:>10} {n_pairs:>12} {cpu_time:>12.2f} "
        row += f"{bf_time:>12.2f} {cl_time:>12.2f} {speedup:>9.1f}x"
        print(row)

    print()


def run_density(n_atoms=10000, cutoff=5.0):
    """Benchmark across different densities."""
    if not HAS_CUPY:
        print("CuPy not available")
        return

    print("=" * 80)
    print(f"Density Benchmark (n_atoms={n_atoms}, cutoff={cutoff})")
    print("=" * 80)
    print()
    header = f"{'Density':>10} {'N pairs':>12} {'CPU (ms)':>12} "
    header += f"{'GPU BF (ms)':>12} {'GPU CL (ms)':>12} {'Speedup':>10}"
    print(header)
    print("-" * 72)

    for density in [0.01, 0.02, 0.05, 0.1, 0.2]:
        positions, box = generate_system(n_atoms, density)
        positions_gpu = cp.asarray(positions)
        box_gpu = cp.asarray(box)

        cpu_time, _, n_pairs = benchmark_cpu(positions, box, cutoff)
        bf_time, _, _ = benchmark_gpu(positions_gpu, box_gpu, cutoff, "brute_force")
        cl_time, _, _ = benchmark_gpu(positions_gpu, box_gpu, cutoff, "cell_list")

        speedup = cpu_time / cl_time
        row = f"{density:>10.2f} {n_pairs:>12} {cpu_time:>12.2f} "
        row += f"{bf_time:>12.2f} {cl_time:>12.2f} {speedup:>9.1f}x"
        print(row)

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vesin Benchmark")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--density", action="store_true", help="Run density benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--n-atoms", type=int, default=50000, help="Number of atoms")
    parser.add_argument("--atom-density", type=float, default=0.05, help="Atom density")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff distance")
    args = parser.parse_args()

    if args.all:
        run_single(args.n_atoms, args.atom_density, args.cutoff)
        print()
        run_scaling(args.atom_density, args.cutoff)
        run_density(args.n_atoms, args.cutoff)
    elif args.scaling:
        run_scaling(args.atom_density, args.cutoff)
    elif args.density:
        run_density(args.n_atoms, args.cutoff)
    else:
        run_single(args.n_atoms, args.atom_density, args.cutoff)
