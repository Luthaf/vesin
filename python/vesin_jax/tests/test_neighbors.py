import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _utils import SYSTEMS, enable_x64, pairs_set, reference
from vesin_jax import NeighborList


@pytest.mark.parametrize("system", SYSTEMS.values(), ids=str)
@pytest.mark.parametrize("cutoff", [3.0, 6.5])
@pytest.mark.parametrize("full_list", [True, False])
def test_neighbors(system, cutoff, full_list):
    expected, expected_distances = reference(system, cutoff, full_list)
    n_expected = len(expected)
    max_pairs = n_expected + 17

    calculator = NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S, d, D, n_pairs = calculator.compute(
        *system.arrays(), system.periodic, "ijSdD", max_pairs=max_pairs
    )

    assert int(n_pairs) == n_expected
    assert i.shape == (max_pairs,)
    assert i.dtype == jnp.int32
    assert S.dtype == jnp.int32
    assert d.dtype == jnp.float32

    assert pairs_set(i, j, S, n_pairs, full_list) == expected
    pairs = slice(None, n_expected)
    np.testing.assert_allclose(
        np.sort(d[pairs]), np.sort(expected_distances), atol=1e-4
    )
    np.testing.assert_allclose(np.linalg.norm(D[pairs], axis=1), d[pairs], atol=1e-5)

    padding = slice(n_expected, None)
    n_points = len(system.points)
    np.testing.assert_equal(np.asarray(i[padding]), n_points)
    np.testing.assert_equal(np.asarray(j[padding]), n_points)
    np.testing.assert_equal(np.asarray(S[padding]), 0)
    np.testing.assert_equal(np.asarray(d[padding]), 0.0)
    np.testing.assert_equal(np.asarray(D[padding]), 0.0)


def test_float64():
    system = SYSTEMS["diamond"]
    expected, expected_distances = reference(system, 4.0, True)

    calculator = NeighborList(cutoff=4.0, full_list=True)
    with enable_x64():
        i, j, S, d, n_pairs = calculator.compute(
            *system.arrays(jnp.float64),
            system.periodic,
            "ijSd",
            max_pairs=len(expected),
        )

    assert d.dtype == jnp.float64
    assert i.dtype == jnp.int32
    assert pairs_set(i, j, S, n_pairs) == expected
    np.testing.assert_allclose(np.sort(d), np.sort(expected_distances), rtol=1e-12)


def test_quantities():
    calculator = NeighborList(cutoff=5.0, full_list=True, sorted=True)
    points, box = SYSTEMS["random-periodic"].arrays()

    i, j, P, n_pairs = calculator.compute(points, box, True, "ijP", max_pairs=2048)
    np.testing.assert_equal(np.asarray(P), np.stack([i, j], axis=1))
    # sorted by `i`, with the padding (`i = n_points`) staying at the end
    assert np.all(np.diff(np.asarray(i)) >= 0)

    (only_n_pairs,) = calculator.compute(points, box, True, "", max_pairs=2048)
    assert int(only_n_pairs) == int(n_pairs)


def test_overflow():
    system = SYSTEMS["random-periodic"]
    n_expected = len(reference(system, 5.0, True)[0])
    n_points = len(system.points)

    calculator = NeighborList(cutoff=5.0, full_list=True)
    for max_pairs in [0, 1, n_expected - 1]:
        i, S, d, n_pairs = calculator.compute(
            *system.arrays(), True, "iSd", max_pairs=max_pairs
        )
        # the actual number of pairs is reported, and all entries are padding
        assert int(n_pairs) == n_expected
        np.testing.assert_equal(np.asarray(i), n_points)
        np.testing.assert_equal(np.asarray(S), 0)
        np.testing.assert_equal(np.asarray(d), 0.0)


def test_no_pairs():
    calculator = NeighborList(cutoff=1.0, full_list=True)
    points = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    i, d, n_pairs = calculator.compute(
        points, jnp.zeros((3, 3)), False, "id", max_pairs=4
    )

    assert int(n_pairs) == 0
    np.testing.assert_equal(np.asarray(i), 2)
    np.testing.assert_equal(np.asarray(d), 0.0)


def test_no_points():
    calculator = NeighborList(cutoff=1.0, full_list=True)

    def distances_sum(points):
        d, n_pairs = calculator.compute(points, jnp.eye(3), True, "d", max_pairs=4)
        return jnp.sum(d), n_pairs

    points = jnp.zeros((0, 3))
    (total, n_pairs), forces = jax.jit(jax.value_and_grad(distances_sum, has_aux=True))(
        points
    )
    assert int(n_pairs) == 0
    assert float(total) == 0.0
    assert forces.shape == (0, 3)


def test_periodic_as_array():
    system = SYSTEMS["random-slab"]
    calculator = NeighborList(cutoff=5.0, full_list=True)

    # the periodic boundary conditions can change without re-compiling the code
    @jax.jit
    def compute(points, box, periodic):
        return calculator.compute(points, box, periodic, "ijS", max_pairs=4096)

    points, box = system.arrays()
    slab = compute(points, box, jnp.array(system.periodic))
    assert pairs_set(*slab) == reference(system, 5.0, True)[0]

    *_, n_pairs_periodic = compute(points, box, jnp.array([True, True, True]))
    assert int(n_pairs_periodic) > int(slab[-1])


def test_errors():
    calculator = NeighborList(cutoff=3.0, full_list=True)
    points = jnp.zeros((4, 3))
    box = jnp.eye(3)

    with pytest.raises(ValueError, match="unexpected character in `quantities`: x"):
        calculator.compute(points, box, True, "ijx", max_pairs=4)

    with pytest.raises(ValueError, match="`max_pairs` must be positive or zero"):
        calculator.compute(points, box, True, max_pairs=-1)

    with pytest.raises(TypeError):
        calculator.compute(points, box, True, max_pairs=4.5)

    message = r"`points` must be an array of shape \(n_points, 3\)"
    with pytest.raises(ValueError, match=message):
        calculator.compute(jnp.zeros((4, 2)), box, True, max_pairs=4)

    with pytest.raises(ValueError, match=r"`box` must be an array of shape \(3, 3\)"):
        calculator.compute(points, jnp.eye(2), True, max_pairs=4)

    with pytest.raises(TypeError, match="`points` must contain floating point values"):
        calculator.compute(jnp.zeros((4, 3), dtype=jnp.int32), box, True, max_pairs=4)

    with pytest.raises(TypeError, match="`points` and `box` must have the same dtype"):
        calculator.compute(points, jnp.eye(3, dtype=jnp.float16), True, max_pairs=4)

    with pytest.raises(ValueError, match="`algorithm` must be one of"):
        NeighborList(cutoff=3.0, full_list=True, algorithm="nope")

    with pytest.raises(
        ValueError, match="n_threads must be zero or a positive integer"
    ):
        NeighborList(cutoff=3.0, full_list=True, n_threads=-1)

    with pytest.raises(ValueError, match="skin must be a finite number"):
        NeighborList(cutoff=3.0, full_list=True, skin=float("nan"))

    if jax.default_backend() == "cpu":
        # errors from vesin itself are raised when running the calculation
        calculator = NeighborList(cutoff=3.0, full_list=True, algorithm="brute_force")
        with pytest.raises(
            Exception, match="only VesinAutoAlgorithm and VesinCellList"
        ):
            jax.block_until_ready(calculator.compute(points, box, True, max_pairs=4))
