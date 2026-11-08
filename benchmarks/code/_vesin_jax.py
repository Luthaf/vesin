from ._base import NeighborList


class VesinJaxNeighborList(NeighborList):
    name = "vesin-jax"
    devices = ("cpu", "cuda")
    run_by_default = False

    @classmethod
    def version(cls):
        import vesin_jax

        return vesin_jax.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        import jax
        import jax.numpy as jnp

        import vesin_jax

        super().__init__(atoms, cutoff, full_list, device)

        # use the same precision as the other implementations
        jax.config.update("jax_enable_x64", True)

        jax_device = jax.devices("gpu" if device == "cuda" else "cpu")[0]
        self.positions = jax.device_put(jnp.asarray(atoms.positions), jax_device)
        self.cell = jax.device_put(jnp.asarray(atoms.cell[:]), jax_device)
        self.pbc = jax.device_put(jnp.asarray(atoms.pbc), jax_device)

        calculator = vesin_jax.NeighborList(cutoff=cutoff, full_list=full_list)

        # count the pairs to pick the capacity, with some room as a molecular
        # dynamics driver would do
        (n_pairs,) = calculator.compute(
            self.positions, self.cell, self.pbc, "", max_pairs=0
        )
        max_pairs = int(1.1 * int(n_pairs)) + 1

        self.compute = jax.jit(
            lambda points, box, periodic: calculator.compute(
                points, box, periodic, quantities="ijSd", max_pairs=max_pairs
            )
        )

    def run(self):
        import jax

        *_, n_pairs = jax.block_until_ready(
            self.compute(self.positions, self.cell, self.pbc)
        )
        return int(n_pairs)
