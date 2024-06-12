import os
import time

import ase.io
import ase.neighborlist
import matscipy.neighbours

import vesin

HERE = os.path.dirname(os.path.realpath(__file__))


def benchmark(function, atoms, cutoff):
    n_warm = 5
    start = time.time()
    for _ in range(n_warm):
        function("ijS", atoms, cutoff)
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
        function("ijS", atoms, cutoff)
    end = time.time()

    return (end - start) / n_iter


atoms = ase.io.read(f"{HERE}/carbon.xyz")
cutoff = 5
max_cell_repeat = 7

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


for kx, ky, kz in repeats:
    super_cell = atoms.repeat((kx, ky, kz))
    print(len(super_cell), "atoms")

    timing = benchmark(ase.neighborlist.neighbor_list, super_cell, cutoff)
    print(f"   ase took {timing * 1e3:.3f} ms")

    timing = benchmark(matscipy.neighbours.neighbour_list, super_cell, float(cutoff))
    print(f"   matscipy took {timing * 1e3:.3f} ms")

    timing = benchmark(vesin.ase_neighbor_list, super_cell, float(cutoff))
    print(f"   vesin took {timing * 1e3:.3f} ms")
    print()
