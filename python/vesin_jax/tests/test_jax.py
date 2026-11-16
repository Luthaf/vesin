import jax
import jax.numpy as jnp
import pytest
from vesin_jax import NeighborList


@pytest.fixture(autouse=True, params=[True, False], ids=["x64", "x32"])
def x64(request):
    """Run all tests both with and without 64-bit arrays in JAX"""
    with jax.enable_x64(request.param):
        yield request.param


@pytest.fixture
def points(x64):
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-6.0, 0.0, 0.0],
            [-6.0, -2.0, 0.0],
            [-6.0, 0.0, -2.0],
        ],
        dtype=jnp.float64 if x64 else jnp.float32,
    )


@pytest.fixture
def box(x64):
    return jnp.diag(
        jnp.array([54.0, 54.0, 54.0], dtype=jnp.float64 if x64 else jnp.float32)
    )


# options giving a static number of pairs, compatible with jax.jit and jax.vmap
STATIC_OPTIONS = [{"padded_to": 3}, {"fixed_n_pairs": 20}]
STATIC_OPTIONS_IDS = ["padded_to", "fixed_n_pairs"]

# all the ways of computing the neighbor list, including with a dynamic number of pairs
ALL_OPTIONS = STATIC_OPTIONS + [{}]
ALL_OPTIONS_IDS = STATIC_OPTIONS_IDS + ["dynamic"]

DEVICES = jax.devices("cpu")
try:
    DEVICES += jax.devices("cuda")
except RuntimeError:
    # no cuda available
    pass


@pytest.mark.parametrize("options", STATIC_OPTIONS, ids=STATIC_OPTIONS_IDS)
def test_jit(points, box, options):
    calculator = NeighborList(cutoff=2.1, full_list=True, **options)

    def function(points, box):
        i, d = calculator.compute(points, box, periodic=True, quantities="id")
        return d.sum()

    expected = function(points, box)
    actual = jax.jit(function)(points, box)

    assert jnp.allclose(actual, expected)


def test_jit_dynamic_fails(points, box):
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=None)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    compiled = jax.jit(function)
    message = (
        "the number of pairs in the neighbor list is only known after the calculation, "
        "which is not compatible with jax.jit, or jax.vmap. Set `padded_to` to a fixed "
        "number of pairs per point, or `fixed_n_pairs` to a fixed total "
        "number of pairs to use these functions"
    )
    with pytest.raises(RuntimeError, match=message):
        compiled(points, box)


@pytest.mark.parametrize("options", STATIC_OPTIONS, ids=STATIC_OPTIONS_IDS)
def test_vmap(points, box, options):
    calculator = NeighborList(cutoff=2.1, full_list=True, **options)

    def function(points, box):
        return calculator.compute(points, box, periodic=True, quantities="PSd")

    # three systems with different neighbors: the reference one, one where all points
    # are too far apart, and one with some pairs crossing the periodic boundary
    all_points = jnp.stack([points, 1.2 * points, points])
    all_boxes = jnp.stack([box, box, 7.0 * jnp.eye(3, dtype=box.dtype)])

    pairs, shifts, distances = jax.vmap(function)(all_points, all_boxes)
    if "padded_to" in options:
        assert pairs.shape == (3, 18, 2)
    else:
        assert pairs.shape == (3, 20, 2)

    n_pairs = (pairs[:, :, 1] >= 0).sum(axis=1)
    assert jnp.array_equal(n_pairs, jnp.array([8, 0, 10]))

    for s in range(3):
        ref_pairs, ref_shifts, ref_distances = function(all_points[s], all_boxes[s])
        assert jnp.array_equal(pairs[s], ref_pairs)
        assert jnp.array_equal(shifts[s], ref_shifts)
        assert jnp.allclose(distances[s], ref_distances)

    # the same box for all systems
    pairs, shifts, distances = jax.vmap(function, in_axes=(0, None))(all_points, box)
    for s in range(3):
        ref_pairs, ref_shifts, ref_distances = function(all_points[s], box)
        assert jnp.array_equal(pairs[s], ref_pairs)
        assert jnp.array_equal(shifts[s], ref_shifts)
        assert jnp.allclose(distances[s], ref_distances)

    # gradients for each system
    def energy(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    grad_function = jax.grad(energy, argnums=(0, 1))
    grad_points, grad_box = jax.vmap(grad_function)(all_points, all_boxes)
    for s in range(3):
        expected = grad_function(all_points[s], all_boxes[s])
        expected_grad_points, expected_grad_box = expected
        assert jnp.allclose(grad_points[s], expected_grad_points)
        assert jnp.allclose(grad_box[s], expected_grad_box)


def test_vmap_dynamic_fails(points, box):
    all_points = jnp.stack([points, 1.2 * points, points])
    all_boxes = jnp.stack([box, box, 7.0 * jnp.eye(3, dtype=box.dtype)])

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=None)

    def function(points, box):
        return calculator.compute(points, box, periodic=True, quantities="PSd")

    message = (
        "the number of pairs in the neighbor list is only known after the calculation, "
        "which is not compatible with jax.jit, or jax.vmap. Set `padded_to` to a fixed "
        "number of pairs per point, or `fixed_n_pairs` to a fixed total "
        "number of pairs to use these functions"
    )
    with pytest.raises(RuntimeError, match=message):
        jax.vmap(function)(all_points, all_boxes)


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=ALL_OPTIONS_IDS)
def test_grad(options, points, box):
    calculator = NeighborList(cutoff=2.1, full_list=True, **options)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    grad_function = jax.grad(function, argnums=(0, 1))
    grad_points, grad_box = grad_function(points, box)

    (vectors,) = calculator.compute(points, box, periodic=True, quantities="D")
    pairs, shifts = calculator.compute(points, box, periodic=True, quantities="PS")

    norm = jnp.linalg.norm(vectors, axis=1)
    unit = vectors / jnp.where(norm > 0, norm, 1.0)[:, None]

    expected_grad_points = jnp.zeros_like(points)
    for pair, u in zip(pairs, unit, strict=True):
        i, j = int(pair[0]), int(pair[1])
        if j >= 0:
            expected_grad_points = expected_grad_points.at[j].add(u)
            expected_grad_points = expected_grad_points.at[i].add(-u)

    assert jnp.allclose(grad_points, expected_grad_points)

    expected_grad_box = shifts.astype(box.dtype).T @ unit
    assert jnp.allclose(grad_box, expected_grad_box)


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=ALL_OPTIONS_IDS)
def test_hessian(options, points):
    # small box, so that the (0, 3) and (3, 0) pairs cross the periodic boundary and
    # the derivatives with respect to the box are not zero
    box = 7.0 * jnp.eye(3, dtype=points.dtype)

    calculator = NeighborList(cutoff=2.1, full_list=True, **options)

    def function(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    hessian = jax.hessian(function, argnums=(0, 1))(points, box)

    # reference calculation, keeping the pairs and shifts fixed
    reference = NeighborList(cutoff=2.1, full_list=True)
    i, j, S = reference.compute(points, box, periodic=True, quantities="ijS")

    def fixed_pairs_function(points, box):
        vectors = points[j] - points[i] + S.astype(box.dtype) @ box
        return jnp.linalg.vector_norm(vectors, axis=1).sum()

    expected = jax.hessian(fixed_pairs_function, argnums=(0, 1))(points, box)

    for actual_block, expected_block in zip(
        jax.tree.leaves(hessian), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.allclose(actual_block, expected_block)


def sort_pairs(pairs, shifts, distances):
    """
    Sort the pairs by ``(i, j, shift)`` on CPU, to compare neighbor lists computed
    with different orders
    """
    cpu = jax.devices("cpu")[0]
    pairs, shifts, distances = jax.device_put((pairs, shifts, distances), cpu)
    order = jnp.lexsort(
        (shifts[:, 2], shifts[:, 1], shifts[:, 0], pairs[:, 1], pairs[:, 0])
    )
    return pairs[order], shifts[order], distances[order]


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("options", ALL_OPTIONS, ids=ALL_OPTIONS_IDS)
def test_devices(device, options, points):
    cpu = jax.devices("cpu")[0]
    # small box, so that some pairs cross the periodic boundary
    box = 7.0 * jnp.eye(3, dtype=points.dtype)

    # the same calculator is used on all devices
    calculator = NeighborList(cutoff=2.1, full_list=True, **options)

    def function(points, box):
        return calculator.compute(points, box, periodic=True, quantities="PSd")

    def energy(points, box):
        (distances,) = calculator.compute(points, box, periodic=True, quantities="d")
        return distances.sum()

    points_cpu = jax.device_put(points, cpu)
    box_cpu = jax.device_put(box, cpu)
    expected_pairs, expected_shifts, expected_distances = sort_pairs(
        *function(points_cpu, box_cpu)
    )
    expected_grad_points, expected_grad_box = jax.grad(energy, argnums=(0, 1))(
        points_cpu, box_cpu
    )

    points = jax.device_put(points, device)
    box = jax.device_put(box, device)

    pairs, shifts, distances = function(points, box)
    for array in (pairs, shifts, distances):
        assert array.devices() == {device}

    pairs, shifts, distances = sort_pairs(pairs, shifts, distances)
    assert jnp.array_equal(pairs, expected_pairs)
    assert jnp.array_equal(shifts, expected_shifts)
    assert jnp.allclose(distances, expected_distances)

    grad_points, grad_box = jax.grad(energy, argnums=(0, 1))(points, box)
    assert grad_points.devices() == {device}
    assert jnp.allclose(jax.device_put(grad_points, cpu), expected_grad_points)
    assert jnp.allclose(jax.device_put(grad_box, cpu), expected_grad_box)

    if options:
        pairs, shifts, distances = jax.jit(function)(points, box)
        assert pairs.devices() == {device}

        pairs, shifts, distances = sort_pairs(pairs, shifts, distances)
        assert jnp.array_equal(pairs, expected_pairs)
        assert jnp.array_equal(shifts, expected_shifts)
        assert jnp.allclose(distances, expected_distances)
