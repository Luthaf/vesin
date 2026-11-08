import functools
import operator
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import batching, mlir, xla

from ._c_lib import FFI_TARGET


# The primitive computing the neighbor list from the points, box and periodic
# flags. It returns the pairs `int32[max_pairs, 2]`, their periodic shifts
# `int32[max_pairs, 3]` and the number of pairs in the neighbor list `int32[]`.
# Only the first `n_pairs` entries of the pairs and shifts are valid, and none
# of them if `n_pairs > max_pairs`. This primitive is not differentiable.
_neighbors_p = Primitive("vesin_neighbors")
_neighbors_p.multiple_results = True
_neighbors_p.def_impl(functools.partial(xla.apply_primitive, _neighbors_p))


@_neighbors_p.def_abstract_eval
def _neighbors_abstract_eval(points, box, periodic, *, max_pairs, **kwargs):
    return (
        jax.core.ShapedArray((max_pairs, 2), np.int32),
        jax.core.ShapedArray((max_pairs, 3), np.int32),
        jax.core.ShapedArray((), np.int32),
    )


def _tensor_type(shape, dtype):
    return mlir.ir.RankedTensorType.get(
        list(shape), mlir.dtype_to_ir_type(np.dtype(dtype))
    )


def _neighbors_lowering(
    ctx,
    points,
    box,
    periodic,
    *,
    max_pairs,
    cutoff,
    full_list,
    sorted,
    algorithm,
    n_threads,
):
    # vesin uses float64 points and returns the pairs as `size_t`. XLA supports
    # these types even when jax runs without `jax_enable_x64`, so we convert
    # from and to the types seen by users inside the compiled code.
    points_aval, box_aval, _ = ctx.avals_in
    if points_aval.dtype != np.float64:
        points = hlo.convert(_tensor_type(points_aval.shape, np.float64), points)
        box = hlo.convert(_tensor_type(box_aval.shape, np.float64), box)

    lowering = jax.ffi.ffi_lowering(
        FFI_TARGET,
        result_types=[
            _tensor_type((max_pairs, 2), np.uint64),
            _tensor_type((max_pairs, 3), np.int32),
            _tensor_type((), np.int32),
        ],
    )

    pairs, shifts, n_pairs = lowering(
        ctx,
        points,
        box,
        periodic,
        cutoff=np.float64(cutoff),
        full_list=bool(full_list),
        sorted=bool(sorted),
        algorithm=algorithm,
        n_threads=np.int64(n_threads),
    )

    pairs = hlo.convert(_tensor_type((max_pairs, 2), np.int32), pairs)
    return [pairs, shifts, n_pairs]


mlir.register_lowering(_neighbors_p, _neighbors_lowering, platform="cpu")
mlir.register_lowering(_neighbors_p, _neighbors_lowering, platform="cuda")


def _neighbors_batching(args, dims, **params):
    # the systems are computed one after the other
    args = [
        jnp.moveaxis(a, d, 0) if d is not None else a
        for a, d in zip(args, dims, strict=True)
    ]

    def single_system(batched):
        batched = iter(batched)
        system = [
            next(batched) if d is not None else a
            for a, d in zip(args, dims, strict=True)
        ]
        return _neighbors_p.bind(*system, **params)

    batched = [a for a, d in zip(args, dims, strict=True) if d is not None]
    outputs = jax.lax.map(single_system, batched)
    return outputs, (0, 0, 0)


batching.primitive_batchers[_neighbors_p] = _neighbors_batching


class NeighborList:
    """
    A neighbor list calculator for JAX. The calculation can be used inside
    :py:func:`jax.jit` and :py:func:`jax.vmap`, and the distances and vectors
    are differentiable with respect to the points and the box.

    The arrays in JAX must have a size known when compiling the code, so the
    neighbor list is returned in arrays with a fixed capacity of ``max_pairs``
    pairs, together with the actual number of pairs ``n_pairs``. Code using the
    neighbor list should check that ``n_pairs <= max_pairs``, and run the
    calculation again with a larger capacity otherwise:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from vesin_jax import NeighborList
    >>> calculator = NeighborList(cutoff=1.5, full_list=True)
    >>> points = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> box = jnp.eye(3) * 10.0
    >>> @jax.jit
    ... def energy(points, box):
    ...     i, j, d, n_pairs = calculator.compute(
    ...         points, box, periodic=True, quantities="ijd", max_pairs=16
    ...     )
    ...     # `d` is zero for the padding entries
    ...     return jnp.sum(jnp.where(d > 0, 1.0 / d, 0.0)), n_pairs
    >>> e, n_pairs = energy(points, box)
    >>> int(n_pairs)
    6
    """

    def __init__(
        self,
        cutoff: float,
        full_list: bool,
        sorted: bool = False,
        algorithm: str = "auto",
        n_threads: int = 0,
    ):
        """
        :param cutoff: spherical cutoff for this neighbor list
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``) or
            only once
        :param sorted: Should the neighbor list be sorted? If ``True``, the returned
            pairs will be sorted by the first point index (``i``). The order of the
            second point index (``j``) and shifts in the list of pairs is unspecified.
        :param algorithm: algorithm to use when computing the neighbor list. One of
            ``"auto"``, ``"brute_force"`` (only available on GPU), or ``"cell_list"``.
        :param n_threads: number of CPU threads to use. Must be 0 or a positive integer.
            If set to 0, Vesin uses ``OMP_NUM_THREADS`` when set to a positive integer,
            and otherwise the number of available CPU cores.
        """
        if algorithm not in ["auto", "brute_force", "cell_list"]:
            raise ValueError(
                "`algorithm` must be one of 'auto', 'brute_force' or 'cell_list', "
                f"got '{algorithm}'"
            )

        n_threads = operator.index(n_threads)
        if n_threads < 0:
            raise ValueError("n_threads must be zero or a positive integer")

        self._cutoff = float(cutoff)
        self._full_list = bool(full_list)
        self._sorted = bool(sorted)
        self._algorithm = str(algorithm)
        self._n_threads = n_threads

    @property
    def cutoff(self) -> float:
        """The cutoff distance for this neighbor list."""
        return self._cutoff

    @property
    def full_list(self) -> bool:
        """Whether this neighbor list returns both ``i-j`` and ``j-i`` pairs."""
        return self._full_list

    @property
    def sorted(self) -> bool:
        """Whether this neighbor list is sorted by the first index ``i``."""
        return self._sorted

    @property
    def algorithm(self) -> str:
        """Algorithm used to compute this neighbor list."""
        return self._algorithm

    @property
    def n_threads(self) -> int:
        """Number of CPU threads requested for the calculation."""
        return self._n_threads

    def compute(
        self,
        points: jax.Array,
        box: jax.Array,
        periodic,
        quantities: str = "ij",
        *,
        max_pairs: int,
    ) -> Tuple[jax.Array, ...]:
        """
        Compute the neighbor list for the system defined by ``points``, ``box``, and
        ``periodic``; returning the requested ``quantities`` and the number of pairs.

        ``quantities`` can contain any combination of the following values:

        - ``"i"`` to get the index of the first point in the pair
        - ``"j"`` to get the index of the second point in the pair
        - ``"P"`` to get the indexes of the two points in the pair simultaneously
        - ``"S"`` to get the periodic shift of the pair
        - ``"d"`` to get the distance between points in the pair
        - ``"D"`` to get the distance vector between points in the pair

        All the returned arrays contain ``max_pairs`` entries. Only the first
        ``n_pairs`` entries correspond to actual pairs, the remaining ones are
        padding: their indexes are set to ``n_points`` (which is out of bounds, and
        ignored by e.g. :py:func:`jax.ops.segment_sum` or ``array.at[...].add()``)
        and their shifts, distances and vectors are set to zero. If the neighbor list
        contains more than ``max_pairs`` pairs (i.e. ``n_pairs > max_pairs``), *all*
        the entries are padding, and the calculation should be run again with a
        larger ``max_pairs``.

        The distances and vectors are computed from the points and box with JAX
        operations, and can be differentiated with respect to both. The indexes and
        shifts are always ``int32`` arrays, and the distances and vectors use the
        same dtype as ``points``.

        :param points: positions of all points in the system, as an array of shape
            ``(n_points, 3)``
        :param box: bounding box of the system, as an array of shape ``(3, 3)``
            containing the box vectors as rows
        :param periodic: should we use periodic boundary conditions? This can be a
            single boolean to enable/disable periodic boundary conditions in all
            directions, or an array of three booleans (one for each direction).
        :param quantities: quantities to return, defaults to "ij"
        :param max_pairs: number of pairs that the returned arrays can hold. This
            must be a Python integer, since it determines the shape of the outputs.
            If it is zero, only ``n_pairs`` is computed, which can be used to find
            the required capacity.

        :return: tuple of arrays as indicated by ``quantities``, followed by the
            number of pairs in the neighbor list, ``n_pairs``, as a scalar ``int32``
            array.
        """
        max_pairs = operator.index(max_pairs)
        if max_pairs < 0:
            raise ValueError(f"`max_pairs` must be positive or zero, got {max_pairs}")

        for q in quantities:
            if q not in "ijPSdD":
                raise ValueError(f"unexpected character in `quantities`: {q}")

        points = jnp.asarray(points)
        box = jnp.asarray(box)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"`points` must be an array of shape (n_points, 3), got {points.shape}"
            )

        if box.shape != (3, 3):
            raise ValueError(f"`box` must be an array of shape (3, 3), got {box.shape}")

        if not jnp.issubdtype(points.dtype, jnp.floating):
            raise TypeError(
                f"`points` must contain floating point values, got {points.dtype}"
            )

        if box.dtype != points.dtype:
            raise TypeError(
                "`points` and `box` must have the same dtype, "
                f"got {points.dtype} and {box.dtype}"
            )

        periodic = jnp.broadcast_to(jnp.asarray(periodic, dtype=bool), (3,))
        n_points = points.shape[0]

        if n_points == 0:
            pairs = jnp.zeros((max_pairs, 2), dtype=jnp.int32)
            shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)
            n_pairs = jnp.zeros((), dtype=jnp.int32)
        else:
            pairs, shifts, n_pairs = _neighbors_p.bind(
                jax.lax.stop_gradient(points),
                jax.lax.stop_gradient(box),
                periodic,
                max_pairs=max_pairs,
                cutoff=self._cutoff,
                full_list=self._full_list,
                sorted=self._sorted,
                algorithm=self._algorithm,
                n_threads=self._n_threads,
            )

        # entries after `n_pairs` (or all of them when the arrays are too small)
        # are padding, which we set to known values
        valid = (jnp.arange(max_pairs) < n_pairs) & (n_pairs <= max_pairs)
        pairs = jnp.where(valid[:, None], pairs, n_points)
        shifts = jnp.where(valid[:, None], shifts, 0)

        if "d" in quantities or "D" in quantities:
            if n_points == 0:
                vectors = jnp.zeros((max_pairs, 3), dtype=points.dtype)
            else:
                # re-compute the vectors with differentiable operations, using
                # in-bounds indexes for the padding entries
                first = jnp.where(valid, pairs[:, 0], 0)
                second = jnp.where(valid, pairs[:, 1], 0)
                cell_shifts = jnp.matmul(
                    shifts.astype(points.dtype),
                    box,
                    precision=jax.lax.Precision.HIGHEST,
                )
                vectors = points[second] - points[first] + cell_shifts
                vectors = jnp.where(valid[:, None], vectors, 0.0)

            # the double `where` avoids NaN in the gradient of the padding
            squared = jnp.sum(vectors**2, axis=1)
            distances = jnp.where(valid, jnp.sqrt(jnp.where(valid, squared, 1.0)), 0.0)

        results = []
        for q in quantities:
            if q == "i":
                results.append(pairs[:, 0])
            elif q == "j":
                results.append(pairs[:, 1])
            elif q == "P":
                results.append(pairs)
            elif q == "S":
                results.append(shifts)
            elif q == "d":
                results.append(distances)
            elif q == "D":
                results.append(vectors)

        results.append(n_pairs)
        return tuple(results)
