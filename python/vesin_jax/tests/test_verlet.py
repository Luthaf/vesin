import gc

import jax
import jax.numpy as jnp
import numpy as np

from _utils import SYSTEMS, pairs_set
from vesin_jax import NeighborList


CUTOFF = 4.0
MAX_PAIRS = 2048


def trajectory():
    """Points and boxes along a random walk, including steps where a point moves
    by more than skin / 2, and where the box changes"""
    system = SYSTEMS["random-periodic"]
    rng = np.random.default_rng(3)
    points = system.points
    box = system.box

    all_points = []
    all_boxes = []
    for step in range(12):
        points = points + rng.normal(scale=0.03, size=points.shape)
        if step == 5:
            points[3] += [0.6, 0.0, 0.0]
        if step == 8:
            box = box * 1.01
        all_points.append(points)
        all_boxes.append(box)

    return (
        jnp.asarray(np.array(all_points), dtype=jnp.float32),
        jnp.asarray(np.array(all_boxes), dtype=jnp.float32),
    )


def test_skin_gives_exact_list():
    exact = NeighborList(cutoff=CUTOFF, full_list=True)
    cached = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)
    points, boxes = trajectory()

    def compute(calculator, points, box):
        return calculator.compute(points, box, True, "ijS", max_pairs=MAX_PAIRS)

    expected = [
        pairs_set(*compute(exact, p, b)) for p, b in zip(points, boxes, strict=True)
    ]

    # the cache is kept across calls to the compiled code
    compute_cached = jax.jit(lambda p, b: compute(cached, p, b))
    for p, b, pairs in zip(points, boxes, expected, strict=True):
        assert pairs_set(*compute_cached(p, b)) == pairs

    # and across steps of a loop inside the compiled code
    def step(carry, frame):
        return carry, compute(cached, *frame)

    _, outputs = jax.jit(lambda frames: jax.lax.scan(step, None, frames))(
        (points, boxes)
    )
    for index, pairs in enumerate(expected):
        assert pairs_set(*(x[index] for x in outputs)) == pairs


def test_shared_cache():
    # using the same NeighborList for different systems rebuilds the cache, but
    # gives the right results
    exact = NeighborList(cutoff=CUTOFF, full_list=True)
    cached = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)

    systems = [SYSTEMS["random-periodic"], SYSTEMS["random-slab"]]
    for system in systems + systems:
        points, box = system.arrays()
        periodic = jnp.array(system.periodic)
        got = cached.compute(points, box, periodic, "ijS", max_pairs=MAX_PAIRS)
        expected = exact.compute(points, box, periodic, "ijS", max_pairs=MAX_PAIRS)
        assert pairs_set(*got) == pairs_set(*expected)

    points = jnp.stack([system.arrays()[0] for system in systems])
    boxes = jnp.stack([system.arrays()[1] for system in systems])

    def compute(calculator):
        return jax.vmap(
            lambda p, b: calculator.compute(p, b, True, "ijS", max_pairs=MAX_PAIRS)
        )(points, boxes)

    got = compute(cached)
    expected = compute(exact)
    for index in range(len(systems)):
        assert pairs_set(*(x[index] for x in got)) == pairs_set(
            *(x[index] for x in expected)
        )


def test_freed_state():
    points, box = SYSTEMS["random-periodic"].arrays()
    expected = pairs_set(
        *NeighborList(cutoff=CUTOFF, full_list=True).compute(
            points, box, True, "ijS", max_pairs=MAX_PAIRS
        )
    )

    # this NeighborList only exists while tracing, and its state is freed before
    # the compiled code runs
    @jax.jit
    def compute(points, box):
        calculator = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)
        return calculator.compute(points, box, True, "ijS", max_pairs=MAX_PAIRS)

    compute.lower(points, box)
    gc.collect()
    for _ in range(3):
        assert pairs_set(*compute(points, box)) == expected
