import contextlib
import os
from dataclasses import dataclass

import ase.io
import jax
import jax.numpy as jnp
import numpy as np

import vesin


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@dataclass
class System:
    name: str
    points: np.ndarray
    box: np.ndarray
    periodic: tuple

    def __str__(self):
        return self.name

    def arrays(self, dtype=jnp.float32):
        """``points`` and ``box`` as JAX arrays"""
        return jnp.asarray(self.points, dtype=dtype), jnp.asarray(self.box, dtype=dtype)


def _from_xyz(name):
    atoms = ase.io.read(os.path.join(CURRENT_DIR, "data", f"{name}.xyz"))
    return System(name, atoms.positions, atoms.cell[:], tuple(atoms.pbc.tolist()))


def _random(name, periodic, seed):
    """40 random points in a triclinic box"""
    rng = np.random.default_rng(seed)
    box = 8.0 * np.eye(3) + rng.uniform(-1.0, 1.0, size=(3, 3))
    points = rng.uniform(0.0, 1.0, size=(40, 3)) @ box
    return System(name, points, box, periodic)


SYSTEMS = {
    system.name: system
    for system in [
        _from_xyz("water"),
        _from_xyz("diamond"),
        _from_xyz("carbon"),
        _from_xyz("Cd2I4O12"),
        _random("random-periodic", (True, True, True), 0),
        _random("random-slab", (True, True, False), 1),
        _random("random-wire", (False, True, False), 2),
    ]
}


@contextlib.contextmanager
def enable_x64():
    """Enable 64-bit types in jax, for the duration of a ``with`` block"""
    if hasattr(jax, "enable_x64"):
        context = jax.enable_x64(True)
    else:
        # older versions of jax
        from jax.experimental import enable_x64

        context = enable_x64()

    with context:
        yield


def canonical_pairs(i, j, S, full_list):
    """Set of ``(i, j, Sx, Sy, Sz)`` tuples. For half lists, each pair is
    canonicalized, since either ``(i, j, S)`` or ``(j, i, -S)`` can be returned."""
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


def pairs_set(i, j, S, n_pairs, full_list=True):
    """Set of pairs in the outputs of ``vesin_jax.NeighborList.compute``. The order
    of pairs is not specified, and can change between calculations on GPU."""
    n = int(n_pairs)
    assert n <= i.shape[0]
    return canonical_pairs(*map(np.asarray, (i[:n], j[:n], S[:n])), full_list)


def reference(system, cutoff, full_list):
    """Set of pairs and distances computed by vesin with numpy"""
    calculator = vesin.NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S, d = calculator.compute(
        system.points, system.box, system.periodic, quantities="ijSd"
    )
    return canonical_pairs(i, j, S, full_list), d
