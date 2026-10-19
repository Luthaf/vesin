import jax
import jax.numpy as jnp
import numpy as np
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


def test_errors():
    points = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=jnp.float64)
    box = jnp.zeros((3, 3), dtype=jnp.float64)

    calculator = NeighborList(cutoff=2.8, full_list=True)

    message = "unexpected character in `quantities`: Q"
    with pytest.raises(ValueError, match=message):
        calculator.compute(points, box, periodic=False, quantities="ijQ")

    message = "`points` and `box` must have the same dtype"
    with pytest.raises(RuntimeError, match=message):
        calculator.compute(points, box.astype(jnp.float32), periodic=False)

    message = "n_threads must be zero or a positive integer"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=2.8, full_list=True, n_threads=-1)

    message = "Unknown algorithm 'foo'"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=2.8, full_list=True, algorithm="foo")

    message = "`padded_to != None` requires `full_list = true`"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=2.0, full_list=False, padded_to=1)

    message = "`padded_to` must be positive or None, got -1"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=1.0, full_list=True, padded_to=-1)


@pytest.mark.parametrize("quantities", ["ijS", "D", "d", "ijSDd"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_no_neighbors(quantities, dtype):
    points = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype)
    box = jnp.eye(3, dtype=dtype)

    calculator = NeighborList(cutoff=0.1, full_list=True)
    outputs = calculator.compute(points, box, True, quantities)

    assert "i" not in quantities or list(outputs[quantities.index("i")].shape) == [0]
    assert "j" not in quantities or list(outputs[quantities.index("j")].shape) == [0]
    assert "S" not in quantities or list(outputs[quantities.index("S")].shape) == [0, 3]
    assert "D" not in quantities or list(outputs[quantities.index("D")].shape) == [0, 3]
    assert "d" not in quantities or list(outputs[quantities.index("d")].shape) == [0]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype(dtype):
    box = jnp.eye(3, dtype=dtype) * 3.0
    points = jax.random.uniform(
        jax.random.key(0), (100, 3), dtype=dtype, minval=0.0, maxval=3.0
    )

    calculator = NeighborList(cutoff=1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, True, "ijSDd")

    assert i.dtype == jnp.int64
    assert j.dtype == jnp.int64
    assert s.dtype == jnp.int32
    assert D.dtype == dtype
    assert d.dtype == dtype


def test_neighbors():
    calculator = NeighborList(cutoff=2.1, full_list=True)
    i, j, S, D, d = calculator.compute(POINTS, BOX, True, "ijSDd")

    assert len(i) == 8
    assert sorted(map(tuple, np.asarray(np.stack([i, j], axis=1)))) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
        (3, 4),
        (3, 5),
        (4, 3),
        (5, 3),
    ]

    assert np.allclose(d, np.linalg.norm(np.asarray(D), axis=1))


def test_pad_neighbors():
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)
    i, j, S, D, d = calculator.compute(POINTS, BOX, True, "ijSDd")

    assert len(i) == 18
    assert len(j) == 18

    counts = np.bincount(np.asarray(i), minlength=6)
    assert np.all(counts == 3)

    padding = np.asarray(j) == -1
    assert int(padding.sum()) == 10
    assert np.all(np.asarray(S)[padding] == 0)
    assert np.all(np.asarray(D)[padding] == 0)
    assert np.all(np.asarray(d)[padding] == 0)

    # non-padded pairs should match the unpadded calculation
    reference = NeighborList(cutoff=2.1, full_list=True)
    ref_i, ref_j, ref_S, ref_D, ref_d = reference.compute(POINTS, BOX, True, "ijSDd")

    keep = ~padding
    assert np.all(np.asarray(i)[keep] == np.asarray(ref_i))
    assert np.all(np.asarray(j)[keep] == np.asarray(ref_j))
    assert np.all(np.asarray(S)[keep] == np.asarray(ref_S))
    assert np.allclose(np.asarray(D)[keep], np.asarray(ref_D))
    assert np.allclose(np.asarray(d)[keep], np.asarray(ref_d))


def test_half_list():
    calculator = NeighborList(cutoff=2.1, full_list=True)
    i, j = calculator.compute(POINTS, BOX, True, "ij")

    assert len(i) == 8

    half = NeighborList(cutoff=2.1, full_list=False)
    hi, hj = half.compute(POINTS, BOX, True, "ij")

    assert len(hi) == 4
    half_pairs = {(int(a), int(b)) for a, b in zip(hi, hj, strict=True)}
    all_pairs = {(int(x), int(y)) for x, y in zip(i, j, strict=True)}
    for a, b in half_pairs:
        assert (a, b) in all_pairs
