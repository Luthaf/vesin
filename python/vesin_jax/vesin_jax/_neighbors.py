from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ._c_lib import get_library
from ._ffi import vesin_jax_compute, vesin_jax_compute_dynamic, vesin_jax_copy_out


_ALGORITHMS = {
    "auto": 0,  # VesinAutoAlgorithm
    "brute_force": 1,  # VesinBruteForce
    "cell_list": 2,  # VesinCellList
}


class NeighborList:
    """
    A neighbor list calculator that can be used with JAX.

    The calculation is compatible with :py:func:`jax.jit` and :py:func:`jax.vmap` when
    either ``padded_to`` or ``fixed_n_pairs`` is set, since the number of pairs
    is then a known, static shape:

    - with ``padded_to``, there are ``n_points * padded_to`` pairs, with missing pairs
      for each point padded with pseudo-pairs ``(i, -1)`` carrying a zero shift;
    - with ``fixed_n_pairs``, there are ``fixed_n_pairs`` pairs, the
      actual pairs coming first, followed by pseudo-pairs ``(-1, -1)`` carrying a zero
      shift.

    If both are ``None``, the number of pairs is dynamic, and the calculation only works
    outside of :py:func:`jax.jit` and :py:func:`jax.vmap`.
    """

    def __init__(
        self,
        *,
        cutoff: float,
        full_list: bool,
        sorted: bool = False,
        padded_to: Optional[int] = None,
        fixed_n_pairs: Optional[int] = None,
        algorithm: str = "auto",
        skin: float = 0.0,
        n_threads: int = 0,
    ):
        """
        :param cutoff: spherical cutoff for this neighbor list
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``) or
            only once
        :param sorted: Should the neighbor list be sorted? If ``True``, the returned
            pairs will be sorted by the first point index (``i``). The order of the
            second point index (``j``) and shifts in the list of pairs is unspecified.
        :param padded_to: pad full lists so every point has exactly this many outgoing
            pairs, using ``(i, -1)`` pseudo-pairs with zero shift, distance and vector.
            Computing the list fails if a point has more neighbors than this. If set to
            ``None``, no padding is applied, and the number of pairs is only known after
            the calculation, which is not compatible with ``jax.jit``.
        :param fixed_n_pairs: pad the neighbor list to exactly this many pairs
            in total, adding ``(-1, -1)`` pseudo-pairs with zero shift, distance and
            vector after the actual pairs. Computing the list fails if there are more
            pairs than this. This can not be used together with ``padded_to``.
        :param algorithm: algorithm to use when computing the neighbor list. One of
            ``"auto"``, ``"brute_force"``, or ``"cell_list"``.
        :param skin: skin size for Verlet caching. A positive value enables caching the
            neighbor list until any atom moves farther than ``skin/2`` from its
            reference coordinates. A negative value sets the skin to an unspecified
            fraction of the cutoff that is optimal for the current implementation. Zero
            disables caching (the default).
        :param n_threads: number of CPU threads to use. Must be 0 or a positive integer.
            If set to 0, Vesin uses ``OMP_NUM_THREADS`` when set to a positive integer,
            and otherwise the number of available CPU cores.
        """
        if padded_to is not None:
            padded_to = int(padded_to)
            if padded_to <= 0:
                raise ValueError(
                    f"`padded_to` must be positive or None, got {padded_to}"
                )

            if not full_list:
                raise ValueError("`padded_to != None` requires `full_list = true`")

        if fixed_n_pairs is not None:
            fixed_n_pairs = int(fixed_n_pairs)
            if fixed_n_pairs <= 0:
                raise ValueError(
                    f"`fixed_n_pairs` must be positive or None, got {fixed_n_pairs}"
                )

            if padded_to is not None:
                raise ValueError(
                    "`padded_to` and `fixed_n_pairs` can not be used together"
                )

        if int(n_threads) < 0:
            raise ValueError("n_threads must be zero or a positive integer")

        algorithm = str(algorithm).lower()
        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                "Must be one of 'auto', 'brute_force', or 'cell_list'."
            )

        self._cutoff = float(cutoff)
        self._full_list = bool(full_list)
        self._sorted = bool(sorted)
        self._padded_to = padded_to if padded_to is not None else 0
        self._fixed_n_pairs = fixed_n_pairs if fixed_n_pairs is not None else 0
        self._algorithm = algorithm
        self._skin = float(skin)
        self._n_threads = int(n_threads)

        self._library = get_library()
        self._state = self._library.vesin_jax_state_create()
        if not self._state:
            raise RuntimeError("failed to allocate vesin-jax state")

    def __del__(self):
        if hasattr(self, "_library") and hasattr(self, "_state"):
            self._library.vesin_jax_state_free(self._state)

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
    def padded_to(self) -> Optional[int]:
        """Number of outgoing pairs each point is padded to, or ``None``."""
        if self._padded_to == 0:
            return None
        return self._padded_to

    @property
    def fixed_n_pairs(self) -> Optional[int]:
        """Total number of pairs the neighbor list is padded to, or ``None``."""
        if self._fixed_n_pairs == 0:
            return None
        return self._fixed_n_pairs

    @property
    def algorithm(self) -> str:
        """Algorithm used to compute this neighbor list."""
        return self._algorithm

    @property
    def skin(self) -> float:
        """Skin size used for Verlet caching."""
        return self._skin

    @property
    def n_threads(self) -> int:
        """Number of CPU threads requested for the calculation."""
        return self._n_threads

    def compute(
        self,
        points: jax.Array,
        box: jax.Array,
        periodic: Union[bool, Sequence[bool]],
        quantities: str = "ij",
    ) -> Tuple[jax.Array, ...]:
        """
        Compute the neighbor list for the system defined by ``points``, ``box``, and
        ``periodic``; returning the requested ``quantities``.

        ``quantities`` can contain any combination of the following values:

        - ``"i"`` to get the index of the first point in the pair
        - ``"j"`` to get the index of the second point in the pair
        - ``"P"`` to get the indexes of the two points in the pair simultaneously
        - ``"S"`` to get the periodic shift of the pair
        - ``"d"`` to get the distance between points in the pair
        - ``"D"`` to get the distance vector between points in the pair

        The distances and distance vectors are always recomputed from the pairs and
        shifts with JAX operations, making the calculation differentiable with respect
        to ``points`` and ``box``, and are returned with the same dtype as ``points``.

        The pairs indexes are returned as ``int64`` arrays if 64-bit support is enabled
        in JAX (``jax.config.update("jax_enable_x64", True)``), and ``int32`` arrays
        otherwise. The shifts are always returned as ``int32`` arrays. The neighbor list
        itself is always computed with ``float64`` precision.

        :param points: positions of all points in the system
        :param box: bounding box of the system, as a 3x3 matrix. Ignored if the system
            is not periodic, but must still be provided.
        :param periodic: should we use periodic boundary conditions? This can be a
            single boolean to enable/disable periodic boundary conditions in all
            directions, or a sequence of three booleans (one for each direction).
        :param quantities: quantities to return, defaults to "ij"

        :return: tuple of :py:class:`jax.Array` as indicated by ``quantities``.
        """
        for q in quantities:
            if q not in "ijPSdD":
                raise ValueError(f"unexpected character in `quantities`: {q}")

        points = jnp.asarray(points)
        box = jnp.asarray(box)
        if box.dtype != points.dtype:
            raise RuntimeError(
                "`points` and `box` must have the same dtype, "
                f"got {points.dtype} and {box.dtype}"
            )

        if isinstance(periodic, bool):
            periodic = (periodic, periodic, periodic)

        if len(periodic) != 3:
            raise ValueError(
                "`periodic` must be a single boolean or a sequence of three booleans"
            )
        periodic_x, periodic_y, periodic_z = (bool(p) for p in periodic)

        options = (
            self._state,
            periodic_x,
            periodic_y,
            periodic_z,
            self._cutoff,
            self._full_list,
            self._sorted,
            self._padded_to,
            self._fixed_n_pairs,
            self._skin,
            _ALGORITHMS[self._algorithm],
            self._n_threads,
        )

        # the C++ code accepts float32 and float64 arrays, and converts them to
        # float64 if needed. Other floating point types are given as float32.
        if points.dtype in (jnp.float32, jnp.float64):
            ffi_points, ffi_box = points, box
        else:
            ffi_points, ffi_box = points.astype(jnp.float32), box.astype(jnp.float32)

        if self._padded_to != 0 or self._fixed_n_pairs != 0:
            pairs, shifts = vesin_jax_compute(ffi_points, ffi_box, options)
        else:
            n_pairs = vesin_jax_compute_dynamic(ffi_points, ffi_box, options)
            pairs, shifts = vesin_jax_copy_out(self._state, n_pairs)

        pairs = jnp.asarray(pairs)
        shifts = jnp.asarray(shifts)

        # pseudo-pairs `(i, -1)` and `(-1, -1)` should have a zero shift, distance
        # and vector; this also prevents out-of-bounds indexing below
        mask = pairs[:, 1] >= 0
        i_index = jnp.where(mask, pairs[:, 0], 0)
        j_index = jnp.where(mask, pairs[:, 1], 0)

        # r_ij = r_j - r_i + S @ H
        vectors = points[j_index] - points[i_index] + shifts.astype(box.dtype) @ box
        vectors = jnp.where(mask[:, None], vectors, 0.0)
        distances = jnp.linalg.vector_norm(vectors, axis=1)

        results: list = []
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

        return tuple(results)
