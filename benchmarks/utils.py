import sys
import time
from functools import partial

from .code import get_implementation


def benchmark(neighbor_list, atoms, cutoff, full_list):
    nl = neighbor_list(atoms, cutoff, full_list)

    # the first run will contain any JIT compilation timing,
    # and is excluded from the warmup.
    n_pairs = int(nl.run())

    # measure the time taken by a couple of warming up iterations
    # to decide how long to run the benchmark for
    start = time.time()
    nl.run()
    end = time.time()
    warmup = end - start

    if warmup < 0.01:
        # do a couple more runs if warmup took less than 10ms
        n_warm = 10
        start = time.time()
        for _ in range(n_warm):
            nl.run()
        end = time.time()

        warmup = (end - start) / n_warm

    # dynamically pick the number of iterations to keep timing below 1s per test, while
    # also ensuring at least 3 repetitions
    n_iter = int(1.0 / warmup)
    if n_iter > 10000:
        n_iter = 10000
    elif n_iter < 3:
        n_iter = 3

    start = time.time()
    for _ in range(n_iter):
        nl.run()
    end = time.time()

    return (end - start) / n_iter, n_pairs


def run_benchmark(impl, device, atoms, cutoff, full_list):
    neighbor_list = partial(get_implementation(impl), device=device)

    try:
        return benchmark(neighbor_list, atoms, cutoff, full_list)
    except Exception as e:
        print(f"+++ {impl} failed to run, error: {e}", file=sys.stderr)
        return float("nan"), None
