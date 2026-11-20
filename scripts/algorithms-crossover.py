#!/usr/bin/env python3
"""
Measure where the CUDA cell list starts to beat the CUDA brute force algorithm, to
update ``FALLBACK_CROSSOVER`` in ``vesin/src/cuda/pick_algo.cpp``.

Background
----------

On CUDA, ``algorithm="auto"`` picks between the brute force and the cell list algorithms
from the number of points. The brute force wins for small systems (it is a single kernel
with no setup), and the cell list wins above some number of points, the *crossover*. The
model used by vesin is that the crossover grows linearly with the *reduced cutoff*, i.e.
the cutoff divided by the typical distance between points ``(volume / n_points)^(1/3)``.
The constant is stored as a number of points per unit of reduced cutoff, and the same
value is used for sorted and unsorted output (the brute force has to sort its output
while the cell list produces it already sorted, but sorting is cheap compared to the
brute force itself).

vesin measures this value on each GPU the first time ``"auto"`` is used, and only falls
back to ``FALLBACK_CROSSOVER`` when this measurement fails. The crossover depends on the
hardware, so the default should be a typical value over a few different GPUs, and be
re-measured when the CUDA kernels change.

What this script does
---------------------

For a set of reduced cutoffs, this script builds slightly perturbed simple cubic
lattices of increasing size, and times the brute force and cell list algorithms on them
(with sorted and unsorted output, half neighbor list, only computing the pairs; i.e. the
same setup as the runtime measurement, which only uses unsorted output). The crossover
is where the timings of both algorithms are equal, interpolated in log-log space between
the two sizes around it. The sweep over sizes stops shortly after the cell list became
faster.

This is done for periodic systems (which are used to suggest a value for
``FALLBACK_CROSSOVER``) and for the same lattices without periodic boundary conditions
(to check the volume estimate used for non-periodic systems). The script also checks the
model, by comparing the crossover per unit of reduced cutoff across cutoffs (to check
that it is linear), and between sorted and unsorted output (to check that a single value
works for both).

All timings include the overhead of calling vesin from Python. This is the same for both
algorithms, so it does not move the point where both timings are equal.

How to run it
-------------

This needs a CUDA GPU, a version of vesin built with CUDA support, and ``cupy``. From
the root of the repository:

.. code-block:: bash

    pip install ./python/vesin scripts/algorithms-crossover.py --device 0

Run this on a few different GPUs (e.g. a consumer card, an A100 and an H100), and use
the geometric mean of the suggested value for ``FALLBACK_CROSSOVER``.
"""

import argparse
import math
import statistics
import time

import cupy
import numpy as np

from vesin import NeighborList


# Spacing between points in the lattices, the same as the runtime measurement.
# The results do not depend on it (only the reduced cutoff matters).
SPACING = 2.0

# Reduced cutoffs (cutoff / spacing) to measure. 2.5 is what the runtime
# measurement uses.
REDUCED_CUTOFFS = [1.5, 2.5, 4.0]

# Number of points along each side of the lattices.
SIDES = [6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 36, 40]

# Number of sizes to measure after the cell list became faster than both brute
# force variants, before stopping the sweep.
EXTRA_SIZES = 2

# Time to spend on each timing, in seconds
TIMING_BUDGET = 0.2


def lattice(side, rng):
    """Simple cubic lattice with ``side**3`` points, slightly perturbed"""
    grid = np.arange(side) * SPACING
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
    points = points.reshape(-1, 3)
    points += rng.uniform(-0.1 * SPACING, 0.1 * SPACING, size=points.shape)
    box = np.eye(3) * side * SPACING
    return points, box


def reduced_cutoff(points, box, periodic, cutoff):
    """
    The cutoff divided by the typical distance between points.

    KEEP IN SYNC: this must compute the same volume as ``analyze_box`` in
    ``vesin/src/cuda/kernels/bruteforce.cu`` (only for fully periodic or fully
    non-periodic systems), and the same reduced cutoff as ``auto_use_cell_list``
    in ``vesin/src/cuda/pick_algo.cpp``.
    """
    if periodic:
        volume = abs(np.linalg.det(box))
    else:
        extent = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(np.maximum(extent, cutoff))
    return cutoff / (volume / len(points)) ** (1 / 3)


def time_compute(calculator, points, box, periodic):
    """
    Median time (in seconds) of one call to ``calculator.compute``, after one
    warmup call. This runs at least 5 times, and until ``TIMING_BUDGET`` seconds have
    been spent (up to 100 times).
    """
    calculator.compute(points, box, periodic, quantities="P", copy=False)
    cupy.cuda.Device().synchronize()

    timings = []
    start = time.perf_counter()
    while len(timings) < 5 or (
        time.perf_counter() - start < TIMING_BUDGET and len(timings) < 100
    ):
        call_start = time.perf_counter()
        calculator.compute(points, box, periodic, quantities="P", copy=False)
        cupy.cuda.Device().synchronize()
        timings.append(time.perf_counter() - call_start)

    return statistics.median(timings)


def find_crossover(cell, brute):
    """
    Find where ``cell`` timings become smaller than ``brute`` timings, by
    interpolating ``log(cell / brute)`` linearly between the two sizes around the
    first crossing.

    Returns ``(index, fraction)`` such that ``log(cell / brute)`` is zero at
    ``fraction`` of the way between sizes ``index`` and ``index + 1``, or ``None``
    if there is no crossing in the measured range. Use ``interpolate`` to get
    the value of any quantity measured for each size at this point.
    """
    log_ratios = [math.log(c / b) for c, b in zip(cell, brute, strict=True)]
    for i in range(len(log_ratios) - 1):
        if log_ratios[i] >= 0.0 and log_ratios[i + 1] < 0.0:
            fraction = log_ratios[i] / (log_ratios[i] - log_ratios[i + 1])
            return i, fraction
    return None


def interpolate(values, crossing):
    index, fraction = crossing
    start = math.log(values[index])
    end = math.log(values[index + 1])
    return math.exp(start + fraction * (end - start))


def describe_missing(n_points, cell, brute):
    if cell[0] < brute[0]:
        return f"below {n_points[0]} points"
    return f"above {n_points[-1]} points"


def measure(reduced, periodic, rng):
    """Sweep over the system sizes for one reduced cutoff and periodicity"""
    cutoff = reduced * SPACING

    # KEEP IN SYNC: same options as `time_neighbors` in
    # `vesin/src/cuda/pick_algo.cpp` (half list, only pairs, no Verlet skin)
    calculators = {
        (algorithm, sort_output): NeighborList(
            cutoff=cutoff,
            full_list=False,
            sorted=sort_output,
            algorithm=algorithm,
            skin=0.0,
        )
        for algorithm in ["brute_force", "cell_list"]
        for sort_output in [False, True]
    }

    rows = []
    cell_faster = 0
    for side in SIDES:
        points, box = lattice(side, rng)
        row = {
            "n_points": len(points),
            "reduced_cutoff": reduced_cutoff(points, box, periodic, cutoff),
        }

        d_points = cupy.asarray(points)
        d_box = cupy.asarray(box)
        for (algorithm, sort_output), calculator in calculators.items():
            key = f"{algorithm}{'_sorted' if sort_output else ''}"
            row[key] = time_compute(calculator, d_points, d_box, periodic)
        rows.append(row)

        print(
            f"    {row['n_points']:6d} points: "
            f"brute force {1e6 * row['brute_force']:9.1f} us "
            f"(sorted {1e6 * row['brute_force_sorted']:9.1f} us), "
            f"cell list {1e6 * row['cell_list']:9.1f} us "
            f"(sorted {1e6 * row['cell_list_sorted']:9.1f} us)",
            flush=True,
        )

        if (
            row["cell_list"] < row["brute_force"]
            and row["cell_list_sorted"] < row["brute_force_sorted"]
        ):
            cell_faster += 1
            if cell_faster > EXTRA_SIZES:
                break
        else:
            cell_faster = 0

    n_points = [row["n_points"] for row in rows]
    result = {
        "reduced_cutoff": reduced,
        "periodic": periodic,
    }

    for sort_output in [False, True]:
        suffix = "_sorted" if sort_output else ""
        cell = [row["cell_list" + suffix] for row in rows]
        brute = [row["brute_force" + suffix] for row in rows]
        crossing = find_crossover(cell, brute)

        name = "sorted" if sort_output else "unsorted"
        if crossing is None:
            result[name] = None
            print(f"  {name}: no crossover ({describe_missing(n_points, cell, brute)})")
            continue

        crossover = interpolate(n_points, crossing)
        reduced_at_crossover = interpolate(
            [row["reduced_cutoff"] for row in rows], crossing
        )
        result[name] = {
            "crossover": crossover,
            "reduced_cutoff": reduced_at_crossover,
            "per_reduced_cutoff": crossover / reduced_at_crossover,
        }

        print(
            f"  {name}: crossover at {crossover:.0f} points, i.e. "
            f"{crossover / reduced_at_crossover:.0f} points per unit of "
            f"reduced cutoff (reduced cutoff = {reduced_at_crossover:.2f})"
        )

    return result


def geometric_mean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values))


def main():
    parser = argparse.ArgumentParser(
        description="measure the default CUDA algorithm crossover on the current GPU"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="index of the CUDA device to use"
    )
    args = parser.parse_args()

    cupy.cuda.Device(args.device).use()
    properties = cupy.cuda.runtime.getDeviceProperties(args.device)
    print(f"measuring on {properties['name'].decode()}")

    rng = np.random.default_rng(seed=0xC055)

    configurations = [(reduced, True) for reduced in REDUCED_CUTOFFS]
    configurations += [(reduced, False) for reduced in REDUCED_CUTOFFS]

    results = []
    for reduced, periodic in configurations:
        kind = "periodic" if periodic else "non-periodic"
        print(f"\n{kind} system, reduced cutoff = {reduced}")
        results.append(measure(reduced, periodic, rng))

    # KEEP IN SYNC: this follows the model in `auto_use_cell_list`
    # (vesin/src/cuda/pick_algo.cpp): a crossover proportional to the reduced
    # cutoff, with the same value for sorted and unsorted output.
    periodic_results = [r for r in results if r["periodic"]]
    per_cutoff = {}
    for name in ["unsorted", "sorted"]:
        per_cutoff[name] = [
            r[name]["per_reduced_cutoff"]
            for r in periodic_results
            if r[name] is not None
        ]

    print("\nsummary")
    print("-------")
    for name, values in per_cutoff.items():
        if len(values) > 1:
            # if the linear model is right, these should all be about the same
            print(
                f"{name}: points per unit of reduced cutoff vary by "
                f"{max(values) / min(values):.2f}x across cutoffs "
                "(1.0x if the linear model is exact)"
            )

    for r in periodic_results:
        if r["unsorted"] is not None and r["sorted"] is not None:
            ratio = r["sorted"]["crossover"] / r["unsorted"]["crossover"]
            print(
                f"reduced cutoff = {r['reduced_cutoff']}: sorted crossover is "
                f"{ratio:.2f}x the unsorted one (1.0x if the same value works for both)"
            )

    all_values = per_cutoff["unsorted"] + per_cutoff["sorted"]
    if len(all_values) == 0:
        print("could not measure the crossover, no suggested value")
    else:
        print(
            "suggested value:\n\n"
            "    static constexpr double FALLBACK_CROSSOVER = "
            f"{geometric_mean(all_values):.1f};"
        )


if __name__ == "__main__":
    main()
