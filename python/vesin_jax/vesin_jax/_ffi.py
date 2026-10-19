import jax
import jax.numpy as jnp


def vesin_jax_compute(points, box, options):
    """
    Run ``vesin_jax_compute``, returning the pairs and shifts. ``options`` is a tuple of
    static parameters.
    """
    (
        state,
        periodic_x,
        periodic_y,
        periodic_z,
        cutoff,
        full,
        sorted,
        padded_to,
        skin,
        algorithm,
        n_threads,
    ) = options

    assert padded_to > 0

    n_pairs = points.shape[0] * padded_to
    compute = jax.ffi.ffi_call(
        "vesin_jax_compute",
        (
            jax.ShapeDtypeStruct((n_pairs, 2), jnp.int64),
            jax.ShapeDtypeStruct((n_pairs, 3), jnp.int32),
        ),
        has_side_effect=True,
        vmap_method="sequential",
    )
    pairs, shifts = compute(
        jax.lax.stop_gradient(points),
        jax.lax.stop_gradient(box),
        jnp.array([state], dtype=jnp.int64),
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        periodic_z=periodic_z,
        cutoff=cutoff,
        full=full,
        sorted=sorted,
        padded_to=padded_to,
        skin=skin,
        algorithm=algorithm,
        n_threads=n_threads,
    )
    return pairs, shifts


########################################################################################


def vesin_jax_compute_dynamic(points, box, options):
    """
    Run ``vesin_jax_compute_dynamic``, returning the number of pairs to copy.
    ``options`` is a tuple of static parameters.
    """
    (
        state,
        periodic_x,
        periodic_y,
        periodic_z,
        cutoff,
        full,
        sorted,
        padded_to,
        skin,
        algorithm,
        n_threads,
    ) = options

    assert padded_to == 0

    compute = jax.ffi.ffi_call(
        "vesin_jax_compute_dynamic",
        jax.ShapeDtypeStruct((1,), jnp.int64),
        has_side_effect=True,
        vmap_method="sequential",
    )
    try:
        n_pairs_array = compute(
            jax.lax.stop_gradient(points),
            jax.lax.stop_gradient(box),
            jnp.array([state], dtype=jnp.int64),
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            periodic_z=periodic_z,
            cutoff=cutoff,
            full=full,
            sorted=sorted,
            skin=skin,
            algorithm=algorithm,
            n_threads=n_threads,
        )
        return int(n_pairs_array[0])
    except jax.errors.ConcretizationTypeError:
        raise RuntimeError(
            "the number of pairs in the neighbor list is only known after the "
            "calculation, which is not compatible with jax.jit, or jax.vmap. "
            "Set `padded_to` to a fixed number of pairs per point to use these "
            "functions."
        )


def vesin_jax_copy_out(state, n_pairs):
    copy_out = jax.ffi.ffi_call(
        "vesin_jax_copy_out",
        (
            jax.ShapeDtypeStruct((n_pairs, 2), jnp.int64),
            jax.ShapeDtypeStruct((n_pairs, 3), jnp.int32),
        ),
        has_side_effect=True,
        vmap_method="sequential",
    )
    pairs, shifts = copy_out(jnp.array([state], dtype=jnp.int64))

    return pairs, shifts
