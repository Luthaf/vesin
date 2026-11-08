import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _utils import SYSTEMS, canonical_pairs
from vesin_jax import NeighborList


CUTOFF = 4.0
MAX_PAIRS = 2048


def pairs_set(i, j, S, n_pairs):
    n = int(n_pairs)
    assert n <= MAX_PAIRS
    return canonical_pairs(*map(np.asarray, (i[:n], j[:n], S[:n])), full_list=True)


def trajectory():
    """Points and boxes along a random walk, with some steps moving the points
    by more than the skin, and some deforming the box"""
    system = SYSTEMS[4]
    rng = np.random.default_rng(3)
    points = np.array(system.points)
    box = np.array(system.box)
    frames = []
    for step in range(12):
        points = points + rng.normal(scale=0.03, size=points.shape)
        if step == 5:
            # one point moves by more than skin / 2
            points[3] += [0.6, 0.0, 0.0]
        if step == 8:
            box = box * 1.01
        frames.append(
            (
                jnp.asarray(points, dtype=jnp.float32),
                jnp.asarray(box, dtype=jnp.float32),
            )
        )
    return frames


def test_skin_gives_exact_list():
    exact = NeighborList(cutoff=CUTOFF, full_list=True)
    cached = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)
    assert cached.skin == 0.5

    compute_exact = jax.jit(
        lambda p, b: exact.compute(p, b, True, "ijS", max_pairs=MAX_PAIRS)
    )
    compute_cached = jax.jit(
        lambda p, b: cached.compute(p, b, True, "ijS", max_pairs=MAX_PAIRS)
    )

    for points, box in trajectory():
        assert pairs_set(*compute_cached(points, box)) == pairs_set(
            *compute_exact(points, box)
        )


def test_skin_in_scan():
    exact = NeighborList(cutoff=CUTOFF, full_list=True, sorted=True)
    cached = NeighborList(cutoff=CUTOFF, full_list=True, sorted=True, skin=0.5)

    frames = trajectory()
    points = jnp.stack([p for p, _ in frames])
    boxes = jnp.stack([b for _, b in frames])

    def step(carry, frame):
        return carry, cached.compute(*frame, True, "ijS", max_pairs=MAX_PAIRS)

    _, (i, j, S, n_pairs) = jax.jit(lambda f: jax.lax.scan(step, None, f))(
        (points, boxes)
    )

    for index, (p, b) in enumerate(frames):
        expected = exact.compute(p, b, True, "ijS", max_pairs=MAX_PAIRS)
        from_scan = (i[index], j[index], S[index], n_pairs[index])
        assert pairs_set(*from_scan) == pairs_set(*expected)


def test_shared_cache():
    # the same NeighborList used for different systems is still correct
    exact = NeighborList(cutoff=CUTOFF, full_list=True)
    cached = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)

    first = SYSTEMS[4]
    second = SYSTEMS[5]
    assert len(first.points) == len(second.points)

    for system in [first, second, first, second]:
        points = jnp.asarray(system.points, dtype=jnp.float32)
        box = jnp.asarray(system.box, dtype=jnp.float32)
        periodic = jnp.array(system.periodic)
        got = cached.compute(points, box, periodic, "ijS", max_pairs=MAX_PAIRS)
        expected = exact.compute(points, box, periodic, "ijS", max_pairs=MAX_PAIRS)
        assert pairs_set(*got) == pairs_set(*expected)

    # including with vmap
    points = jnp.stack([jnp.asarray(s.points, dtype=jnp.float32) for s in SYSTEMS[4:7]])
    boxes = jnp.stack([jnp.asarray(s.box, dtype=jnp.float32) for s in SYSTEMS[4:7]])

    def compute(calculator):
        return jax.vmap(
            lambda p, b: calculator.compute(p, b, True, "ijS", max_pairs=MAX_PAIRS)
        )(points, boxes)

    got = compute(cached)
    expected = compute(exact)
    for index in range(len(points)):
        assert pairs_set(*(x[index] for x in got)) == pairs_set(
            *(x[index] for x in expected)
        )


def test_freed_state():
    system = SYSTEMS[4]
    points = jnp.asarray(system.points, dtype=jnp.float32)
    box = jnp.asarray(system.box, dtype=jnp.float32)
    expected = pairs_set(
        *NeighborList(cutoff=CUTOFF, full_list=True).compute(
            points, box, True, "ijS", max_pairs=MAX_PAIRS
        )
    )

    # the NeighborList only lives while tracing, its state is freed before the
    # compiled code runs, which must still work
    @jax.jit
    def compute(points, box):
        calculator = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)
        return calculator.compute(points, box, True, "ijS", max_pairs=MAX_PAIRS)

    compute.lower(points, box)
    gc.collect()
    for _ in range(3):
        assert pairs_set(*compute(points, box)) == expected

    # freeing the state of a NeighborList that was used
    calculator = NeighborList(cutoff=CUTOFF, full_list=True, skin=0.5)
    calculator.compute(points, box, True, "ijS", max_pairs=MAX_PAIRS)
    del calculator
    gc.collect()


def test_invalid_skin():
    with pytest.raises(ValueError, match="skin must be a finite number"):
        NeighborList(cutoff=CUTOFF, full_list=True, skin=float("nan"))
