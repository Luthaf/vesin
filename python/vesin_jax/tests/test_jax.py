import jax
import jax.numpy as jnp
import pytest
from vesin_jax import NeighborList


jax.config.update("jax_enable_x64", True)

POINTS = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [-6.0, 0.0, 0.0],
        [-6.0, -2.0, 0.0],
        [-6.0, 0.0, -2.0],
    ],
    dtype=jnp.float64,
)
BOX = jnp.diag(jnp.array([54.0, 54.0, 54.0], dtype=jnp.float64))


def test_jit():
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    expected = function(POINTS, BOX)
    actual = jax.jit(function)(POINTS, BOX)

    assert jnp.allclose(actual, expected)


def test_jit_dynamic_fails():
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=None)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    compiled = jax.jit(function)
    message = (
        "the number of pairs in the neighbor list is only known after the calculation, "
        "which is not compatible with jax.jit, or jax.vmap. Set `padded_to` to a fixed "
        "number of pairs per point to use these functions"
    )
    with pytest.raises(RuntimeError, match=message):
        compiled(POINTS, BOX)


def test_vmap():
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)

    def function(points, box):
        return calculator.compute(points, box, periodic=True, quantities="PSd")

    # three systems with different neighbors: the reference one, one where all points
    # are too far apart, and one with some pairs crossing the periodic boundary
    points = jnp.stack([POINTS, 1.2 * POINTS, POINTS])
    box = jnp.stack([BOX, BOX, 7.0 * jnp.eye(3, dtype=jnp.float64)])

    pairs, shifts, distances = jax.vmap(function)(points, box)
    assert pairs.shape == (3, 18, 2)

    n_pairs = (pairs[:, :, 1] >= 0).sum(axis=1)
    assert jnp.array_equal(n_pairs, jnp.array([8, 0, 10]))

    for s in range(3):
        ref_pairs, ref_shifts, ref_distances = function(points[s], box[s])
        assert jnp.array_equal(pairs[s], ref_pairs)
        assert jnp.array_equal(shifts[s], ref_shifts)
        assert jnp.allclose(distances[s], ref_distances)

    # the same box for all systems
    pairs, shifts, distances = jax.vmap(function, in_axes=(0, None))(points, BOX)
    for s in range(3):
        ref_pairs, ref_shifts, ref_distances = function(points[s], BOX)
        assert jnp.array_equal(pairs[s], ref_pairs)
        assert jnp.array_equal(shifts[s], ref_shifts)
        assert jnp.allclose(distances[s], ref_distances)

    # gradients for each system
    def energy(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    grad_function = jax.grad(energy, argnums=(0, 1))
    grad_points, grad_box = jax.vmap(grad_function)(points, box)
    for s in range(3):
        expected_grad_points, expected_grad_box = grad_function(points[s], box[s])
        assert jnp.allclose(grad_points[s], expected_grad_points)
        assert jnp.allclose(grad_box[s], expected_grad_box)


def test_vmap_dynamic_fails():
    points = jnp.stack([POINTS, 1.2 * POINTS, POINTS])
    box = jnp.stack([BOX, BOX, 7.0 * jnp.eye(3, dtype=jnp.float64)])

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=None)

    def function(points, box):
        return calculator.compute(points, box, periodic=True, quantities="PSd")

    message = (
        "the number of pairs in the neighbor list is only known after the calculation, "
        "which is not compatible with jax.jit, or jax.vmap. Set `padded_to` to a fixed "
        "number of pairs per point to use these functions"
    )
    with pytest.raises(RuntimeError, match=message):
        jax.vmap(function)(points, box)


@pytest.mark.parametrize("padded_to", [3, None])
def test_grad(padded_to):
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=padded_to)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    grad_function = jax.grad(function, argnums=(0, 1))
    grad_points, grad_box = grad_function(POINTS, BOX)

    (vectors,) = calculator.compute(POINTS, BOX, periodic=True, quantities="D")
    pairs, shifts = calculator.compute(POINTS, BOX, periodic=True, quantities="PS")

    norm = jnp.linalg.norm(vectors, axis=1)
    unit = vectors / jnp.where(norm > 0, norm, 1.0)[:, None]

    expected_grad_points = jnp.zeros_like(POINTS)
    for pair, u in zip(pairs, unit, strict=True):
        i, j = int(pair[0]), int(pair[1])
        if j >= 0:
            expected_grad_points = expected_grad_points.at[j].add(u)
            expected_grad_points = expected_grad_points.at[i].add(-u)

    assert jnp.allclose(grad_points, expected_grad_points)

    expected_grad_box = shifts.astype(jnp.float64).T @ unit
    assert jnp.allclose(grad_box, expected_grad_box)


@pytest.mark.parametrize("padded_to", [3, None])
def test_hessian(padded_to):
    # small box, so that the (0, 3) and (3, 0) pairs cross the periodic boundary and
    # the derivatives with respect to the box are not zero
    box = 7.0 * jnp.eye(3, dtype=jnp.float64)

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=padded_to)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    hessian = jax.hessian(function, argnums=(0, 1))(POINTS, box)

    # reference calculation, keeping the pairs and shifts fixed
    reference = NeighborList(cutoff=2.1, full_list=True)
    i, j, S = reference.compute(POINTS, box, periodic=True, quantities="ijS")

    def fixed_pairs_function(points, box):
        vectors = points[j] - points[i] + S.astype(jnp.float64) @ box
        return jnp.linalg.vector_norm(vectors, axis=1).sum()

    expected = jax.hessian(fixed_pairs_function, argnums=(0, 1))(POINTS, box)

    for actual_block, expected_block in zip(
        jax.tree.leaves(hessian), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.allclose(actual_block, expected_block)
