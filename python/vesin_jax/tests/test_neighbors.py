import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vesin_jax import NeighborList


DEVICES = jax.devices("cpu")
try:
    DEVICES += jax.devices("cuda")
except RuntimeError:
    # no cuda available
    pass


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


def test_errors(points, box):
    calculator = NeighborList(cutoff=2.8, full_list=True)

    message = "unexpected character in `quantities`: Q"
    with pytest.raises(ValueError, match=message):
        calculator.compute(points, box, periodic=False, quantities="ijQ")

    message = "`points` and `box` must have the same dtype"
    with pytest.raises(RuntimeError, match=message):
        calculator.compute(points, box.astype(jnp.float16), periodic=False)

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

    message = "`fixed_n_pairs` must be positive or None, got 0"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=1.0, full_list=True, fixed_n_pairs=0)

    message = "`padded_to` and `fixed_n_pairs` can not be used together"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=1.0, full_list=True, padded_to=3, fixed_n_pairs=10)

    calculator = NeighborList(cutoff=2.1, full_list=True, fixed_n_pairs=1)
    message = (
        "the pre-allocated output arrays can hold 1 pairs, "
        "but this calculation needs 8 pairs"
    )
    with pytest.raises(RuntimeError, match=message):
        calculator.compute(points, box, periodic=False)


@pytest.mark.parametrize("quantities", ["ijS", "D", "d", "ijSDd"])
def test_no_neighbors(quantities):
    points = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=jnp.float32)
    box = jnp.eye(3, dtype=jnp.float32)

    calculator = NeighborList(cutoff=0.1, full_list=True)
    outputs = calculator.compute(points, box, True, quantities)

    assert "i" not in quantities or list(outputs[quantities.index("i")].shape) == [0]
    assert "j" not in quantities or list(outputs[quantities.index("j")].shape) == [0]
    assert "S" not in quantities or list(outputs[quantities.index("S")].shape) == [0, 3]
    assert "D" not in quantities or list(outputs[quantities.index("D")].shape) == [0, 3]
    assert "d" not in quantities or list(outputs[quantities.index("d")].shape) == [0]


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
def test_dtype(dtype, x64):
    if dtype is jnp.float64 and not x64:
        # can not use float64 if x64 is disabled
        return

    box = jnp.eye(3, dtype=dtype) * 3.0
    points = jax.random.uniform(
        jax.random.key(0), (100, 3), dtype=dtype, minval=0.0, maxval=3.0
    )

    calculator = NeighborList(cutoff=1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, True, "ijSDd")

    index_dtype = jnp.int64 if x64 else jnp.int32
    assert i.dtype == index_dtype
    assert j.dtype == index_dtype
    assert s.dtype == jnp.int32
    assert D.dtype == dtype
    assert d.dtype == dtype


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_neighbors(points, box, device):
    points = jax.device_put(points, device)
    box = jax.device_put(box, device)

    calculator = NeighborList(cutoff=2.1, full_list=True)
    i, j, S, D, d = calculator.compute(points, box, True, "ijSDd")

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


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_pad_neighbors(points, box, device):
    points = jax.device_put(points, device)
    box = jax.device_put(box, device)

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)
    i, j, S, D, d = calculator.compute(points, box, True, "ijSDd")

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
    ref_i, ref_j, ref_S, ref_D, ref_d = reference.compute(points, box, True, "ijSDd")

    keep = ~padding
    assert np.all(np.asarray(i)[keep] == np.asarray(ref_i))
    assert np.all(np.asarray(j)[keep] == np.asarray(ref_j))
    assert np.all(np.asarray(S)[keep] == np.asarray(ref_S))
    assert np.allclose(np.asarray(D)[keep], np.asarray(ref_D))
    assert np.allclose(np.asarray(d)[keep], np.asarray(ref_d))


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("full_list", [True, False])
def test_fixed_n_pairs(points, box, device, full_list):
    points = jax.device_put(points, device)
    box = jax.device_put(box, device)

    calculator = NeighborList(cutoff=2.1, full_list=full_list, fixed_n_pairs=20)
    outputs = calculator.compute(points, box, True, "ijSDd")
    for array in outputs:
        assert array.devices() == {device}
    i, j, S, D, d = (np.asarray(array) for array in outputs)

    assert len(i) == 20
    assert len(j) == 20

    # the actual pairs come first, followed by padding
    n_pairs = 8 if full_list else 4
    padding = j == -1
    assert not np.any(padding[:n_pairs])
    assert np.all(padding[n_pairs:])

    assert np.all(i[padding] == -1)
    assert np.all(S[padding] == 0)
    assert np.all(D[padding] == 0)
    assert np.all(d[padding] == 0)

    # actual pairs should match the calculation without padding
    reference = NeighborList(cutoff=2.1, full_list=full_list)
    ref_i, ref_j, ref_S, ref_D, ref_d = (
        np.asarray(array) for array in reference.compute(points, box, True, "ijSDd")
    )

    # the order of pairs is not deterministic on CUDA. There are no periodic
    # images in this box, so `(i, j)` uniquely identifies a pair.
    order = np.argsort(i[:n_pairs] * len(points) + j[:n_pairs])
    ref_order = np.argsort(ref_i * len(points) + ref_j)

    assert np.all(i[:n_pairs][order] == ref_i[ref_order])
    assert np.all(j[:n_pairs][order] == ref_j[ref_order])
    assert np.all(S[:n_pairs][order] == ref_S[ref_order])
    assert np.allclose(D[:n_pairs][order], ref_D[ref_order])
    assert np.allclose(d[:n_pairs][order], ref_d[ref_order])

    # exactly enough room for all pairs
    calculator = NeighborList(cutoff=2.1, full_list=full_list, fixed_n_pairs=n_pairs)
    i, j = (np.asarray(array) for array in calculator.compute(points, box, True, "ij"))
    order = np.argsort(i * len(points) + j)
    assert np.all(i[order] == ref_i[ref_order])
    assert np.all(j[order] == ref_j[ref_order])


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_n_pairs(points, box, x64, device):
    points = jax.device_put(points, device)
    box = jax.device_put(box, device)

    # each point has 2 neighbors, for 8 pairs in total
    def check_n_pairs(n, expected):
        assert isinstance(n, jax.Array)
        assert n.devices() == points.devices()
        assert n.shape == ()
        assert n.dtype == (jnp.int64 if x64 else jnp.int32)
        assert int(n) == expected

    calculator = NeighborList(cutoff=2.1, full_list=True)
    i, n = calculator.compute(points, box, True, "in")
    assert i.shape == (8,)
    check_n_pairs(n, 8)

    calculator = NeighborList(cutoff=2.1, full_list=True, fixed_n_pairs=20)
    i, n = calculator.compute(points, box, True, "in")
    assert i.shape == (20,)
    check_n_pairs(n, 8)

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)
    i, n = calculator.compute(points, box, True, "in")
    assert i.shape == (18,)
    check_n_pairs(n, 18)

    # without enough room, the calculation does not fail and only returns padding
    calculator = NeighborList(cutoff=2.1, full_list=True, fixed_n_pairs=4)
    P, S, D, d, n = calculator.compute(points, box, True, "PSDdn")
    check_n_pairs(n, 8)
    assert np.all(np.asarray(P) == -1)
    assert np.all(np.asarray(S) == 0)
    assert np.all(np.asarray(D) == 0)
    assert np.all(np.asarray(d) == 0)

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=1)
    P, n = calculator.compute(points, box, True, "Pn")
    check_n_pairs(n, 12)
    assert P.shape == (6, 2)
    assert np.all(np.asarray(P) == -1)

    # it still fails if the number of pairs is not requested
    message = "point 0 has more neighbors than the requested `padded_to` \\(1\\)"
    with pytest.raises(RuntimeError, match=message):
        calculator.compute(points, box, True, "P")


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_half_list(points, box, device):
    points = jax.device_put(points, device)
    box = jax.device_put(box, device)

    calculator = NeighborList(cutoff=2.1, full_list=True)
    i, j = calculator.compute(points, box, True, "ij")

    assert len(i) == 8

    half = NeighborList(cutoff=2.1, full_list=False)
    hi, hj = half.compute(points, box, True, "ij")

    assert len(hi) == 4
    half_pairs = {(int(a), int(b)) for a, b in zip(hi, hj, strict=True)}
    all_pairs = {(int(x), int(y)) for x, y in zip(i, j, strict=True)}
    for a, b in half_pairs:
        assert (a, b) in all_pairs
