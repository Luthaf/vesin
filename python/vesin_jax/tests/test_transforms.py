import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from _utils import SYSTEMS, canonical_pairs, reference
from vesin_jax import NeighborList


def pairs_set(i, j, S, n_pairs):
    # the order of pairs with the same `i` is not specified, and can change
    # between runs on GPU
    n = int(n_pairs)
    return canonical_pairs(*map(np.asarray, (i[:n], j[:n], S[:n])), full_list=True)


def test_jit_compiles_once():
    calculator = NeighborList(cutoff=5.0, full_list=True)
    system = SYSTEMS[4]

    n_traces = 0

    @jax.jit
    def compute(points, box):
        nonlocal n_traces
        n_traces += 1
        return calculator.compute(points, box, True, "ijS", max_pairs=2048)

    points = jnp.asarray(system.points, dtype=jnp.float32)
    box = jnp.asarray(system.box, dtype=jnp.float32)

    rng = np.random.default_rng(0)
    for _ in range(3):
        points = points + rng.normal(scale=0.1, size=points.shape).astype(np.float32)
        box = box * 1.01
        i, j, S, n_pairs = compute(points, box)

        moved = dataclasses.replace(
            system,
            points=np.asarray(points, dtype=np.float64),
            box=np.asarray(box, dtype=np.float64),
        )
        expected, _ = reference(moved, 5.0, True)
        assert int(n_pairs) == len(expected)
        assert pairs_set(i, j, S, n_pairs) == expected

    assert n_traces == 1


def test_vmap():
    calculator = NeighborList(cutoff=5.0, full_list=True, sorted=True)
    systems = SYSTEMS[4:7]
    points = jnp.stack([jnp.asarray(s.points, dtype=jnp.float32) for s in systems])
    boxes = jnp.stack([jnp.asarray(s.box, dtype=jnp.float32) for s in systems])
    periodic = jnp.array([s.periodic for s in systems])

    def compute(points, box, periodic):
        return calculator.compute(points, box, periodic, "ijS", max_pairs=2048)

    i, j, S, n_pairs = jax.jit(jax.vmap(compute))(points, boxes, periodic)
    assert n_pairs.shape == (len(systems),)
    for index in range(len(systems)):
        single = compute(points[index], boxes[index], periodic[index])
        batched = (i[index], j[index], S[index], n_pairs[index])
        assert pairs_set(*batched) == pairs_set(*single)

    # some of the inputs are not batched
    i, j, S, n_pairs = jax.vmap(compute, in_axes=(0, None, None))(
        points, boxes[0], periodic[0]
    )
    for index in range(len(systems)):
        single = compute(points[index], boxes[0], periodic[0])
        batched = (i[index], j[index], S[index], n_pairs[index])
        assert pairs_set(*batched) == pairs_set(*single)


def test_scan():
    # a simple MD-like loop, re-computing the neighbor list at each step
    calculator = NeighborList(cutoff=4.0, full_list=True, sorted=True)
    system = SYSTEMS[4]
    points = jnp.asarray(system.points, dtype=jnp.float32)
    box = jnp.asarray(system.box, dtype=jnp.float32)

    rng = np.random.default_rng(1)
    displacements = jnp.asarray(
        rng.normal(scale=0.05, size=(5, *points.shape)), dtype=jnp.float32
    )

    def step(points, displacement):
        points = points + displacement
        outputs = calculator.compute(points, box, True, "ijS", max_pairs=1024)
        return points, outputs

    _, (i, j, S, n_pairs) = jax.jit(lambda p: jax.lax.scan(step, p, displacements))(
        points
    )

    for index in range(len(displacements)):
        points = points + displacements[index]
        single = calculator.compute(points, box, True, "ijS", max_pairs=1024)
        from_scan = (i[index], j[index], S[index], n_pairs[index])
        assert pairs_set(*from_scan) == pairs_set(*single)
        # the list is still sorted by `i`
        np.testing.assert_array_equal(np.asarray(i[index]), np.asarray(single[0]))


def test_inputs_from_previous_computation():
    # the points come from a long computation in the same compiled function,
    # the neighbor list must wait for it to finish (on GPU, this checks that
    # vesin runs on the same CUDA stream as XLA)
    calculator = NeighborList(cutoff=4.0, full_list=True)
    system = SYSTEMS[4]
    points = jnp.asarray(system.points, dtype=jnp.float32)
    box = jnp.asarray(system.box, dtype=jnp.float32)

    rng = np.random.default_rng(2)
    matrix = jnp.asarray(rng.normal(size=(1024, 1024)) / 32, dtype=jnp.float32)

    @jax.jit
    def compute(points, matrix):
        for _ in range(20):
            matrix = jnp.tanh(matrix @ matrix)
        # tiny contribution, only creating a dependency on the loop above
        points = points + 1e-30 * matrix[: points.shape[0], :3]
        return calculator.compute(points, box, True, "ijS", max_pairs=1024)

    expected = pairs_set(*calculator.compute(points, box, True, "ijS", max_pairs=1024))
    for _ in range(10):
        assert pairs_set(*compute(points, matrix)) == expected
