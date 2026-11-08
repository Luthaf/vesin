import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from _utils import SYSTEMS, enable_x64
from vesin_jax import NeighborList


CUTOFF = 4.0


def pair_energy(calculator, points, box, periodic, max_pairs=4096):
    """A pair potential which is smooth at the cutoff"""
    i, d, n_pairs = calculator.compute(points, box, periodic, "id", max_pairs=max_pairs)
    is_pair = i < points.shape[0]
    return jnp.sum(jnp.where(is_pair, (CUTOFF - d) ** 3 * (1.0 + d), 0.0))


@pytest.mark.parametrize(
    "system",
    [
        SYSTEMS[name]
        for name in ["Cd2I4O12", "random-periodic", "random-slab", "random-wire"]
    ],
    ids=str,
)
def test_grad(system):
    calculator = NeighborList(cutoff=CUTOFF, full_list=False)

    with enable_x64():
        points, box = system.arrays(jnp.float64)

        def energy(points, box):
            return pair_energy(calculator, points, box, system.periodic)

        check_grads(energy, (points, box), order=2, modes=("fwd", "rev"), eps=1e-6)
        check_grads(jax.jit(energy), (points, box), order=1, modes=("rev",), eps=1e-6)


def test_vectors_grad():
    calculator = NeighborList(cutoff=CUTOFF, full_list=True)

    with enable_x64():
        points, box = SYSTEMS["random-periodic"].arrays(jnp.float64)

        def vectors_sum(points, box):
            (D, n_pairs) = calculator.compute(points, box, True, "D", max_pairs=4096)
            return jnp.sum(jnp.sin(D))

        check_grads(vectors_sum, (points, box), order=2, modes=("fwd", "rev"), eps=1e-6)


def test_no_nan_in_padding():
    # most of the entries are padding, which must not create NaN in gradients
    calculator = NeighborList(cutoff=CUTOFF, full_list=True)
    points, box = SYSTEMS["random-periodic"].arrays()

    def energy(points, box):
        (d, n_pairs) = calculator.compute(points, box, True, "d", max_pairs=8192)
        return jnp.sum(d)

    forces, virial = jax.jit(jax.grad(energy, argnums=(0, 1)))(points, box)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(virial))

    hessian = jax.hessian(energy)(points, box)
    assert np.all(np.isfinite(hessian))


def test_float32_grad():
    calculator = NeighborList(cutoff=CUTOFF, full_list=True)

    def forces(dtype):
        points, box = SYSTEMS["random-periodic"].arrays(dtype)
        grad = jax.grad(lambda p: pair_energy(calculator, p, box, True))(points)
        assert grad.dtype == dtype
        return np.asarray(grad, dtype=np.float64)

    forces_f32 = forces(jnp.float32)
    with enable_x64():
        forces_f64 = forces(jnp.float64)

    np.testing.assert_allclose(forces_f32, forces_f64, rtol=1e-3, atol=1e-3)
