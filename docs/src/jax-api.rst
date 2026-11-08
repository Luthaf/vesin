.. _jax-api:

JAX API reference
=================

.. autoclass:: vesin_jax.NeighborList
    :members:

Using the neighbor list inside compiled code
--------------------------------------------

The neighbor list runs as an XLA custom call, which is compiled together with
the code using it, on CPU and CUDA devices. Since the shape of all arrays must
be known when compiling the code, the neighbor list is returned in arrays with a
fixed capacity (``max_pairs``), and the code calling it is responsible for
checking that this capacity was large enough. A typical pattern is to return the
number of pairs from the compiled function, and to run it again with a larger
capacity if needed:

.. code-block:: python

    from functools import partial

    import jax
    import jax.numpy as jnp

    from vesin_jax import NeighborList

    calculator = NeighborList(cutoff=5.0, full_list=True)


    @partial(jax.jit, static_argnames="max_pairs")
    def energy(points, box, max_pairs):
        i, d, n_pairs = calculator.compute(
            points, box, True, quantities="id", max_pairs=max_pairs
        )
        # the padding entries have `i == len(points)`, and are ignored by
        # `segment_sum`
        pair_energies = jnp.where(i < len(points), (5.0 - d) ** 2, 0.0)
        per_atom = jax.ops.segment_sum(pair_energies, i, num_segments=len(points))
        return jnp.sum(per_atom), n_pairs


    points = jax.random.uniform(jax.random.key(0), (1000, 3)) * 20.0
    box = 20.0 * jnp.eye(3)

    max_pairs = 1024
    e, n_pairs = energy(points, box, max_pairs)
    if n_pairs > max_pairs:
        # leave some room, to avoid compiling the code again at the next step
        max_pairs = int(1.2 * n_pairs)
        e, n_pairs = energy(points, box, max_pairs)

Calling :py:meth:`NeighborList.compute <vesin_jax.NeighborList.compute>` with
``max_pairs=0`` only computes the number of pairs, which can be used to pick an
initial capacity.

Verlet caching
--------------

With ``skin > 0``, vesin keeps a list of candidate pairs computed with ``cutoff +
skin`` between calls, and only re-computes it when a point moved by more than
``skin / 2`` (or the box changed too much). The returned neighbor list is always
the exact one for ``cutoff``, so this does not change the results, only the cost
of the calculation. This cache lives in vesin's native library, and each
:py:class:`NeighborList <vesin_jax.NeighborList>` has its own: it works best
when one ``NeighborList`` is used for a single system, for example along a
molecular dynamics trajectory.

Gradients
---------

The neighbor list itself (pairs and periodic shifts) is not differentiable, but
the distances and distance vectors are re-computed from the points and box with
JAX operations. They can be differentiated with respect to both the points and
the box (e.g. to compute forces and stress), to any order and in both forward
and reverse mode.

Precision
---------

The neighbor list works with ``float32`` and ``float64`` inputs, and does not
require enabling ``jax_enable_x64``. The pairs are always found with ``float64``
arithmetic, while the distances and vectors are computed in the dtype of the
inputs. With ``float32`` inputs, some distances can thus be slightly above the
cutoff.

GPU
---

On CUDA devices, the calculation runs on the same CUDA stream as the rest of the
XLA computation, and synchronizes this stream once to read the number of pairs
back on the host. Vesin allocates its own GPU memory (outside of the memory pool
of XLA). If you are running out of GPU memory, you might want to reduce the
fraction of memory pre-allocated by XLA with the
``XLA_PYTHON_CLIENT_MEM_FRACTION`` environment variable, or disable the
pre-allocation with ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.

Batches of systems can be computed with :py:func:`jax.vmap`, which runs the
neighbor list calculations one system after the other.
