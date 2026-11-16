import importlib.metadata

import numpy as np

from ._base import NeighborList


class KUPSNeighborList(NeighborList):
    """``AdaptiveNeighborList`` from kUPS, which picks between a dense search and a
    cell list from the number of particles."""

    name = "kups"
    devices = ("cpu", "cuda")
    # kUPS always produces directed edges, i.e. a full neighbor list
    full_list = (True,)

    @classmethod
    def version(cls):
        return importlib.metadata.version("kups")

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        import jax
        import jax.numpy as jnp
        from kups.core.cell import (
            Cell,
            OrthogonalFrame,
            TriclinicFrame,
            to_lower_triangular,
        )
        from kups.core.data import Index, Table
        from kups.core.neighborlist import (
            AdaptiveNeighborList,
            UniversalNeighborlistParameters,
        )
        from kups.core.result import as_result_function
        from kups.core.typing import ParticleId, SystemId
        from kups.core.utils.jax import dataclass

        super().__init__(atoms, cutoff, full_list, device)

        # jax is configured on first use, and every benchmark runs in its own
        # process, so we can select the device and precision globally here.
        jax.config.update("jax_platforms", device)
        jax.config.update("jax_enable_x64", True)

        @dataclass
        class Points:
            positions: jax.Array
            system: Index
            inclusion: Index
            exclusion: Index

        @dataclass
        class Systems:
            cell: Cell

        @dataclass
        class State:
            neighborlist_params: UniversalNeighborlistParameters

        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell[:], dtype=np.float64)
        pbc = tuple(bool(p) for p in atoms.pbc)

        if not any(pbc) and abs(np.linalg.det(cell)) < 1e-6:
            # kUPS still needs a bounding box for fully non-periodic systems
            origin = positions.min(axis=0)
            positions = positions - origin
            frame = OrthogonalFrame(jnp.asarray(positions.max(axis=0) + 1.0)[None])
        else:
            # TriclinicFrame only keeps the lower triangular part of the cell
            # matrix, so rotate the whole system to make the cell lower triangular
            lower, rotate = to_lower_triangular(jnp.asarray(cell))
            positions = rotate(jnp.asarray(positions))
            frame = TriclinicFrame.from_matrix(lower[None])

        n_atoms = len(atoms)
        system_keys = (SystemId(0),)
        system = Index(system_keys, jnp.zeros(n_atoms, dtype=int))
        self.particles = Table(
            tuple(ParticleId(i) for i in range(n_atoms)),
            Points(
                positions=jnp.asarray(positions),
                system=system,
                inclusion=system,
                # each particle is in its own exclusion group, to remove the
                # self-pairs while keeping the pairs with periodic images
                exclusion=Index.integer(jnp.arange(n_atoms)),
            ),
        )
        self.systems = Table(system_keys, Systems(cell=Cell.from_pbc(frame, pbc)))
        cutoffs = Table(system_keys, jnp.array([cutoff]))

        state = State(
            UniversalNeighborlistParameters.estimate(
                Table(system_keys, jnp.array([n_atoms])), self.systems, cutoffs
            )
        )

        def compute(nl, particles, systems):
            edges = nl(particles, systems)
            valid = edges.indices.valid_mask.all(axis=-1)
            vectors = edges.difference_vectors(particles, systems)[:, 0, :]
            distances = jnp.where(valid, jnp.linalg.norm(vectors, axis=-1), 0.0)
            return valid.sum(), distances

        self._compute = jax.jit(as_result_function(compute))

        # The capacities of the neighbor list are fixed at compile time, grow
        # them until they are large enough for this system.
        for _ in range(10):
            self.nl = AdaptiveNeighborList.from_state(state, cutoffs)
            result = self._compute(self.nl, self.particles, self.systems)
            if not result.failed_assertions:
                break
            state = result.fix_or_raise(state)
        result.raise_assertion()

    def run(self):
        n_pairs, distances = self._compute(self.nl, self.particles, self.systems).value
        distances.block_until_ready()
        return int(n_pairs)
