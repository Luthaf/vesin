import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _utils import SYSTEMS, canonical_pairs, enable_x64, reference
from vesin_jax import NeighborList


def compute(calculator, system, dtype, max_pairs, quantities="ijSdD"):
    return calculator.compute(
        jnp.asarray(system.points, dtype=dtype),
        jnp.asarray(system.box, dtype=dtype),
        system.periodic,
        quantities=quantities,
        max_pairs=max_pairs,
    )


@pytest.mark.parametrize("system", SYSTEMS, ids=str)
@pytest.mark.parametrize("cutoff", [3.0, 6.5])
@pytest.mark.parametrize("full_list", [True, False])
def test_consistency(system, cutoff, full_list):
    expected, expected_distances = reference(system, cutoff, full_list)
    n_expected = len(expected)

    calculator = NeighborList(cutoff=cutoff, full_list=full_list)
    max_pairs = n_expected + 17

    i, j, S, d, D, n_pairs = compute(calculator, system, jnp.float32, max_pairs)
    assert int(n_pairs) == n_expected

    assert i.dtype == jnp.int32
    assert S.dtype == jnp.int32
    assert d.dtype == jnp.float32
    assert i.shape == (max_pairs,)
    assert D.shape == (max_pairs, 3)

    valid = slice(0, n_expected)
    i, j, S, d, D = map(np.asarray, (i, j, S, d, D))
    assert canonical_pairs(i[valid], j[valid], S[valid], full_list) == expected
    np.testing.assert_allclose(
        np.sort(d[valid]), np.sort(expected_distances), atol=1e-4
    )
    np.testing.assert_allclose(np.linalg.norm(D[valid], axis=1), d[valid], atol=1e-5)

    # padding
    padding = slice(n_expected, None)
    n_points = len(system.points)
    np.testing.assert_equal(i[padding], n_points)
    np.testing.assert_equal(j[padding], n_points)
    np.testing.assert_equal(S[padding], 0)
    np.testing.assert_equal(d[padding], 0.0)
    np.testing.assert_equal(D[padding], 0.0)


@pytest.mark.parametrize("system", SYSTEMS[:2], ids=str)
def test_float64(system):
    expected, expected_distances = reference(system, 4.0, True)
    calculator = NeighborList(cutoff=4.0, full_list=True)

    with enable_x64():
        i, j, S, d, D, n_pairs = compute(calculator, system, jnp.float64, len(expected))

    assert d.dtype == jnp.float64
    assert i.dtype == jnp.int32
    assert (
        canonical_pairs(np.asarray(i), np.asarray(j), np.asarray(S), True) == expected
    )
    np.testing.assert_allclose(np.sort(d), np.sort(expected_distances), rtol=1e-12)


def test_sorted():
    system = SYSTEMS[4]
    calculator = NeighborList(cutoff=5.0, full_list=True, sorted=True)
    i, n_pairs = compute(calculator, system, jnp.float32, 2048, quantities="i")

    i = np.asarray(i)
    assert int(n_pairs) < 2048
    # padding (with index n_points) stays at the end
    assert np.all(np.diff(i) >= 0)


def test_quantities():
    system = SYSTEMS[4]
    calculator = NeighborList(cutoff=5.0, full_list=True)
    i, j, P, n_pairs = compute(calculator, system, jnp.float32, 2048, quantities="ijP")
    np.testing.assert_equal(np.asarray(P), np.stack([i, j], axis=1))

    outputs = compute(calculator, system, jnp.float32, 2048, quantities="")
    assert len(outputs) == 1
    assert int(outputs[0]) == int(n_pairs)


def test_overflow():
    system = SYSTEMS[4]
    calculator = NeighborList(cutoff=5.0, full_list=True)
    n_expected = len(reference(system, 5.0, True)[0])
    n_points = len(system.points)

    for max_pairs in [0, 1, n_expected - 1]:
        i, j, S, d, n_pairs = compute(
            calculator, system, jnp.float32, max_pairs, "ijSd"
        )
        # the actual number of pairs is reported
        assert int(n_pairs) == n_expected
        # and all the entries are padding
        assert i.shape == (max_pairs,)
        np.testing.assert_equal(np.asarray(i), n_points)
        np.testing.assert_equal(np.asarray(j), n_points)
        np.testing.assert_equal(np.asarray(S), 0)
        np.testing.assert_equal(np.asarray(d), 0.0)

    i, n_pairs = compute(calculator, system, jnp.float32, n_expected, "i")
    assert int(n_pairs) == n_expected
    assert np.all(np.asarray(i) < n_points)


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
    points = jnp.zeros((0, 3))
    i, S, d, D, n_pairs = calculator.compute(
        points, jnp.eye(3), True, "iSdD", max_pairs=4
    )
    assert int(n_pairs) == 0
    np.testing.assert_equal(np.asarray(i), 0)
    np.testing.assert_equal(np.asarray(S), 0)
    np.testing.assert_equal(np.asarray(D), 0.0)

    # also under jit, and with gradients
    def energy(points):
        (d, _) = calculator.compute(points, jnp.eye(3), True, "d", max_pairs=4)
        return jnp.sum(d)

    assert jax.jit(jax.grad(energy))(points).shape == (0, 3)


def test_periodic_as_array():
    system = SYSTEMS[5]
    calculator = NeighborList(cutoff=5.0, full_list=True)
    expected, _ = reference(system, 5.0, True)

    @jax.jit
    def compute_with(points, box, periodic):
        return calculator.compute(points, box, periodic, "ijS", max_pairs=4096)

    points = jnp.asarray(system.points, dtype=jnp.float32)
    box = jnp.asarray(system.box, dtype=jnp.float32)
    i, j, S, n_pairs = compute_with(points, box, jnp.array(system.periodic))
    n = int(n_pairs)
    assert canonical_pairs(*map(np.asarray, (i[:n], j[:n], S[:n])), True) == expected

    # the periodicity is not a compile-time constant
    *_, n_pairs_all = compute_with(points, box, jnp.array([True, True, True]))
    assert int(n_pairs_all) > n


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

    with pytest.raises(
        ValueError, match=r"`points` must be an array of shape \(n_points, 3\)"
    ):
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

    if jax.default_backend() == "cpu":
        # errors from vesin are raised when running the calculation
        calculator = NeighborList(cutoff=3.0, full_list=True, algorithm="brute_force")
        with pytest.raises(
            Exception, match="only VesinAutoAlgorithm and VesinCellList"
        ):
            jax.block_until_ready(calculator.compute(points, box, True, max_pairs=4))
