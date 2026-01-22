#!/usr/bin/env python
"""Benchmark comparison: brute force vs cell list vs CPU.

Generates a plot with 3 subplots for different cutoffs.
"""

import matplotlib.pyplot as plt
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


def benchmark_gpu(algorithm, positions_gpu, box_gpu, cutoff, n_warmup=25, n_runs=50):
    """Benchmark GPU neighbor list computation."""
    nl = NeighborList(cutoff=cutoff, full_list=True, algorithm=algorithm)

    # Warmup
    for _ in range(n_warmup):
        nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    times = []

    for _ in range(n_runs):
        start_event.record()
        nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
        end_event.record()
        end_event.synchronize()
        times.append(cp.cuda.get_elapsed_time(start_event, end_event))

    return np.mean(times), np.std(times)


def benchmark_cpu(positions, box, cutoff, n_warmup=5, n_runs=10):
    """Benchmark CPU neighbor list computation."""
    import time

    nl = NeighborList(cutoff=cutoff, full_list=True)

    # Warmup
    for _ in range(n_warmup):
        nl.compute(positions, box, periodic=True, quantities="ij")

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        nl.compute(positions, box, periodic=True, quantities="ij")
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def run_benchmarks(n_atoms_list, cutoffs, density=0.05, n_warmup=25, n_runs=50):
    """Run benchmarks for all configurations."""
    results = {
        cutoff: {"brute_force": [], "cell_list": [], "cpu": []} for cutoff in cutoffs
    }

    for cutoff in cutoffs:
        print(f"\n=== Cutoff: {cutoff} Å ===")

        for n_atoms in n_atoms_list:
            positions, box = generate_system(n_atoms, density)
            box_size = box[0, 0]

            # Check if cutoff is valid for this box size
            if cutoff > box_size / 2:
                print(
                    f"  {n_atoms:6d} atoms: cutoff too \
                    large for box size {box_size:.1f}"
                )
                results[cutoff]["brute_force"].append((n_atoms, np.nan, np.nan))
                results[cutoff]["cell_list"].append((n_atoms, np.nan, np.nan))
                results[cutoff]["cpu"].append((n_atoms, np.nan, np.nan))
                continue

            positions_gpu = cp.asarray(positions)
            box_gpu = cp.asarray(box)

            print(f"  {n_atoms:6d} atoms (box={box_size:.1f}): ", end="", flush=True)

            # Brute force GPU
            try:
                mean, std = benchmark_gpu(
                    "brute_force", positions_gpu, box_gpu, cutoff, n_warmup, n_runs
                )
                results[cutoff]["brute_force"].append((n_atoms, mean, std))
                print(f"BF={mean:.2f}ms ", end="", flush=True)
            except Exception as e:
                print(f"BF=ERROR({e}) ", end="", flush=True)
                results[cutoff]["brute_force"].append((n_atoms, np.nan, np.nan))

            # Cell list GPU
            try:
                mean, std = benchmark_gpu(
                    "cell_list", positions_gpu, box_gpu, cutoff, n_warmup, n_runs
                )
                results[cutoff]["cell_list"].append((n_atoms, mean, std))
                print(f"CL={mean:.2f}ms ", end="", flush=True)
            except Exception as e:
                print(f"CL=ERROR({e}) ", end="", flush=True)
                results[cutoff]["cell_list"].append((n_atoms, np.nan, np.nan))

            # CPU (fewer runs for large systems)
            cpu_warmup = min(n_warmup, 5)
            cpu_runs = min(n_runs, 10) if n_atoms > 10000 else min(n_runs, 20)
            try:
                mean, std = benchmark_cpu(positions, box, cutoff, cpu_warmup, cpu_runs)
                results[cutoff]["cpu"].append((n_atoms, mean, std))
                print(f"CPU={mean:.2f}ms")
            except Exception as e:
                print(f"CPU=ERROR({e})")
                results[cutoff]["cpu"].append((n_atoms, np.nan, np.nan))

    return results


def plot_results(results, cutoffs, output_file="benchmark_comparison.png"):
    """Create comparison plot with subplots for each cutoff."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    colors = {"brute_force": "C0", "cell_list": "C1", "cpu": "C2"}
    labels = {
        "brute_force": "GPU Brute Force",
        "cell_list": "GPU Cell List",
        "cpu": "CPU",
    }
    markers = {"brute_force": "o", "cell_list": "s", "cpu": "^"}

    for ax, cutoff in zip(axes, cutoffs, strict=True):
        for method in ["brute_force", "cell_list", "cpu"]:
            data = results[cutoff][method]
            n_atoms = [d[0] for d in data]
            means = [d[1] for d in data]
            stds = [d[2] for d in data]

            # Filter out NaN values
            valid = [
                (n, m, s)
                for n, m, s in zip(n_atoms, means, stds, strict=True)
                if not np.isnan(m)
            ]
            if valid:
                n_atoms_valid, means_valid, stds_valid = zip(*valid, strict=True)
                ax.errorbar(
                    n_atoms_valid,
                    means_valid,
                    yerr=stds_valid,
                    label=labels[method],
                    color=colors[method],
                    marker=markers[method],
                    markersize=6,
                    capsize=3,
                    linewidth=1.5,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of atoms")
        ax.set_title(f"Cutoff = {cutoff} Å")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()

    axes[0].set_ylabel("Time (ms)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    # Configuration
    n_points = 10
    n_atoms_list = np.logspace(np.log10(100), np.log10(100000), n_points).astype(int)
    # Remove duplicates and sort
    n_atoms_list = sorted(set(n_atoms_list))
    cutoffs = [3.0, 6.0, 12.0]
    density = 0.05

    print("Benchmark Configuration:")
    print(f"  Atom counts: {n_atoms_list}")
    print(f"  Cutoffs: {cutoffs} Å")
    print(f"  Density: {density}")
    print("  Warmup: 25 iterations")
    print("  Measurement: 50 iterations")

    results = run_benchmarks(n_atoms_list, cutoffs, density, n_warmup=25, n_runs=50)
    plot_results(results, cutoffs)
