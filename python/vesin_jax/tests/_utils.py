import contextlib
import os
from dataclasses import dataclass

import ase.io
import jax
import numpy as np

import vesin


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@dataclass
class System:
    name: str
    points: np.ndarray
    box: np.ndarray
    periodic: tuple

    def __repr__(self):
        return self.name


def _from_xyz(name):
    atoms = ase.io.read(os.path.join(CURRENT_DIR, "data", f"{name}.xyz"))
    return System(
        name=name,
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=tuple(bool(p) for p in atoms.pbc),
    )


def _random(name, periodic, seed):
    rng = np.random.default_rng(seed)
    box = 8.0 * np.eye(3) + rng.uniform(-1.0, 1.0, size=(3, 3))
    points = rng.uniform(0.0, 1.0, size=(40, 3)) @ box
    return System(name=name, points=points, box=box, periodic=periodic)


SYSTEMS = [
    _from_xyz("water"),
    _from_xyz("diamond"),
    _from_xyz("carbon"),
    _from_xyz("Cd2I4O12"),
    _random("random-periodic", (True, True, True), 0),
    _random("random-slab", (True, True, False), 1),
    _random("random-wire", (False, True, False), 2),
]


@contextlib.contextmanager
def enable_x64():
    """Enable 64-bit types in jax, for the duration of a ``with`` block"""
    if hasattr(jax, "enable_x64"):
        context = jax.enable_x64(True)
    else:
        # older versions of jax
        from jax.experimental import enable_x64 as context_manager

        context = context_manager()

    with context:
        yield


def canonical_pairs(i, j, S, full_list):
    """Set of ``(i, j, Sx, Sy, Sz)`` tuples, with half-list pairs canonicalized"""
    pairs = set()
    for pi, pj, (sx, sy, sz) in zip(i.tolist(), j.tolist(), S.tolist(), strict=True):
        if not full_list:
            total = sx + sy + sz
            if pi > pj or (
                pi == pj
                and (total < 0 or (total == 0 and (sz < 0 or (sz == 0 and sy < 0))))
            ):
                pi, pj, sx, sy, sz = pj, pi, -sx, -sy, -sz
        pairs.add((pi, pj, sx, sy, sz))
    return pairs


def reference(system, cutoff, full_list):
    """Reference neighbor list, computed with the numpy version of vesin"""
    calculator = vesin.NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S, d = calculator.compute(
        system.points, system.box, system.periodic, quantities="ijSd"
    )
    return canonical_pairs(i, j, S, full_list), d
