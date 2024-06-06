import ase.build
import ase.io
import ase.neighborlist
import time
import os
import matscipy

import matscipy.neighbours
import vesin

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


atoms = ase.io.read(f"{CURRENT_DIR}/carbon.xyz")
r_cut = 5.0

start = time.time()
for _ in range(1000):
    _ = ase.neighborlist.neighbor_list("ijS", atoms, r_cut)
end = time.time()
print(f"ase NL took {end - start:.3f} ms")

start = time.time()
for _ in range(1000):
    matscipy.neighbours.neighbour_list('ijS', atoms, r_cut)
end = time.time()
print(f"matscipy NL took {end - start:.3f} ms")

start = time.time()
for _ in range(1000):
    _ = vesin.ase_neighbor_list("ijS", atoms, r_cut)
end = time.time()
print(f"vesin NL took {end - start:.3f} ms")
