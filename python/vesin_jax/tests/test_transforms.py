import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from _utils import SYSTEMS, pairs_set, reference
from vesin_jax import NeighborList


def test_jit():
    system = SYSTEMS["random-periodic"]
    calculator = NeighborList(cutoff=5.0, full_list=True)

    n_traces = 0

    @jax.jit
    def compute(points, box):
        nonlocal n_traces
        n_traces += 1
        return calculator.compute(points, box, True, "ijS", max_pairs=2048)

    # new points and box do not require compiling the code again
    rng = np.random.default_rng(0)
    for _ in range(3):
        system = dataclasses.replace(
            system,
            points=system.points + rng.normal(scale=0.1, size=system.points.shape),
            box=system.box * 1.01,
        )
        assert pairs_set(*compute(*system.arrays())) == reference(system, 5.0, True)[0]

    assert n_traces == 1


def test_vmap():
    calculator = NeighborList(cutoff=5.0, full_list=True)
    systems = [
        SYSTEMS["random-periodic"],
        SYSTEMS["random-slab"],
        SYSTEMS["random-wire"],
    ]
    points = jnp.stack([system.arrays()[0] for system in systems])
    boxes = jnp.stack([system.arrays()[1] for system in systems])
    periodic = jnp.array([system.periodic for system in systems])

    def compute(points, box, periodic):
        return calculator.compute(points, box, periodic, "ijS", max_pairs=2048)

    batched = jax.jit(jax.vmap(compute))(points, boxes, periodic)
    for index in range(len(systems)):
        single = compute(points[index], boxes[index], periodic[index])
        assert pairs_set(*(x[index] for x in batched)) == pairs_set(*single)

    # with some inputs not batched
    batched = jax.vmap(compute, in_axes=(0, None, None))(points, boxes[0], periodic[0])
    for index in range(len(systems)):
        single = compute(points[index], boxes[0], periodic[0])
        assert pairs_set(*(x[index] for x in batched)) == pairs_set(*single)


def test_depends_on_previous_operations():
    # The points come from a long computation in the same compiled code, which
    # the neighbor list must wait for. On GPU, this checks that vesin runs on
    # the same CUDA stream as XLA.
    calculator = NeighborList(cutoff=4.0, full_list=True)
    points, box = SYSTEMS["random-periodic"].arrays()
    expected = pairs_set(*calculator.compute(points, box, True, "ijS", max_pairs=1024))

    @jax.jit
    def compute(points, matrix):
        for _ in range(20):
            matrix = jnp.tanh(matrix @ matrix)
        # negligible change of the points, which depends on the loop above
        points = points + 1e-30 * matrix[: points.shape[0], :3]
        return calculator.compute(points, box, True, "ijS", max_pairs=1024)

    rng = np.random.default_rng(1)
    matrix = jnp.asarray(rng.normal(size=(1024, 1024)) / 32, dtype=jnp.float32)
    for _ in range(10):
        assert pairs_set(*compute(points, matrix)) == expected
