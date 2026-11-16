from typing import Any, Dict, List, Optional, Union

import torch


class NeighborList:
    """A neighbor list calculator that can be used with TorchScript."""

    def __init__(
        self,
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
            ``None``, no padding is applied.
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
        n_threads = int(n_threads)
        if n_threads < 0:
            raise ValueError("n_threads must be zero or a positive integer")

        self._cutoff = float(cutoff)
        self._full_list = bool(full_list)
        self._sorted = bool(sorted)
        self._algorithm = str(algorithm)
        self._skin = float(skin)
        self._n_threads = n_threads

        self._padded_to = 0
        if padded_to is not None:
            if padded_to <= 0:
                raise ValueError(
                    f"`padded_to` must be positive or None, got {padded_to}"
                )

            self._padded_to = padded_to
            if not self._full_list:
                raise ValueError("`padded_to != None` requires `full_list = true`")

        self._fixed_n_pairs = 0
        if fixed_n_pairs is not None:
            if fixed_n_pairs <= 0:
                raise ValueError(
                    f"`fixed_n_pairs` must be positive or None, got {fixed_n_pairs}"
                )

            if padded_to is not None:
                raise ValueError(
                    "`padded_to` and `fixed_n_pairs` can not be used together"
                )
            self._fixed_n_pairs = fixed_n_pairs

        self._state = torch.classes.vesin.State()

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

    def __getstate__(self) -> Dict[str, Any]:
        # the state only contains memory allocated by vesin, which is not
        # serialized. We save the options and create a new state when loading
        return {
            "cutoff": self._cutoff,
            "full_list": self._full_list,
            "sorted": self._sorted,
            "padded_to": self._padded_to,
            "fixed_n_pairs": self._fixed_n_pairs,
            "algorithm": self._algorithm,
            "skin": self._skin,
            "n_threads": self._n_threads,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # TorchScript needs `isinstance` to refine the type of the values
        cutoff = state["cutoff"]
        assert isinstance(cutoff, float)
        self._cutoff = cutoff

        full_list = state["full_list"]
        assert isinstance(full_list, bool)
        self._full_list = full_list

        sorted = state["sorted"]
        assert isinstance(sorted, bool)
        self._sorted = sorted

        padded_to = state["padded_to"]
        assert isinstance(padded_to, int)
        self._padded_to = padded_to

        fixed_n_pairs = state["fixed_n_pairs"]
        assert isinstance(fixed_n_pairs, int)
        self._fixed_n_pairs = fixed_n_pairs

        algorithm = state["algorithm"]
        assert isinstance(algorithm, str)
        self._algorithm = algorithm

        skin = state["skin"]
        assert isinstance(skin, float)
        self._skin = skin

        n_threads = state["n_threads"]
        assert isinstance(n_threads, int)
        self._n_threads = n_threads

        self._state = torch.classes.vesin.State()

    def compute(
        self,
        points: torch.Tensor,
        box: torch.Tensor,
        periodic: Union[bool, torch.Tensor],
        quantities: str = "ij",
        copy: bool = True,
    ) -> List[torch.Tensor]:
        """
        Compute the neighbor list for the system defined by ``positions``, ``box``, and
        ``periodic``; returning the requested ``quantities``.

        ``quantities`` can contain any combination of the following values:

        - ``"i"`` to get the index of the first point in the pair
        - ``"j"`` to get the index of the second point in the pair
        - ``"P"`` to get the indexes of the two points in the pair simultaneously
        - ``"S"`` to get the periodic shift of the pair
        - ``"d"`` to get the distance between points in the pair
        - ``"D"`` to get the distance vector between points in the pair

        :param points: positions of all points in the system
        :param box: bounding box of the system
        :param periodic: should we use periodic boundary conditions? This can be a
            single boolean to enable/disable periodic boundary conditions in all
            directions, or a tensor of three booleans (one for each direction).
        :param quantities: quantities to return, defaults to "ij"
        :param copy: should we copy the returned quantities, defaults to ``True``.
            Setting this to ``False`` might be a bit faster, but the returned tensors
            are view inside this class, and will be invalidated whenever this class is
            garbage collected or used to run a new calculation.

        :return: list of :py:class:`torch.Tensor` as indicated by ``quantities``.
        """
        if isinstance(periodic, bool):
            periodic = torch.as_tensor(periodic)

        initial_dtype = points.dtype
        if box.dtype != initial_dtype:
            raise RuntimeError(
                "`points` and `box` must have the same dtype, "
                f"got {points.dtype} and {box.dtype}"
            )

        for q in quantities:
            if q not in "ijPSdD":
                raise ValueError(f"unexpected character in `quantities`: {q}")

        # check the device here, since meta tensors would otherwise be sent to the
        # fake implementation of `vesin::neighbors`
        if points.device != box.device:
            raise ValueError(
                "expected `points` and `box` to have the same device, "
                f"got {points.device} and {box.device}"
            )

        if points.device.type not in ("cpu", "cuda"):
            raise RuntimeError(f"device {points.device} is not supported in vesin")

        return_shifts = "S" in quantities
        return_distances = "d" in quantities
        return_vectors = "D" in quantities

        requires_grad = (
            torch.is_grad_enabled()
            and (points.requires_grad or box.requires_grad)
            and (return_distances or return_vectors)
        )

        points_f64 = points.to(torch.float64)
        box_f64 = box.to(torch.float64)

        if requires_grad:
            # `vesin::neighbors` is not differentiable, so we re-compute the vectors
            # and distances from the pairs and shifts with standard torch operations
            # instead. This makes the calculation differentiable (to any order), and
            # traceable by `torch.compile` and `torch.export`. The shifts are used in
            # the backward pass, and must not be invalidated by the next call, so we
            # always copy them.
            pairs, shifts, _, _ = torch.ops.vesin.neighbors(
                self._state,
                points_f64.detach(),
                box_f64.detach(),
                periodic,
                self._cutoff,
                self._full_list,
                self._sorted,
                self._padded_to,
                self._fixed_n_pairs,
                self._algorithm,
                self._skin,
                self._n_threads,
                True,
                False,
                False,
                True,
            )

            first = pairs[:, 0]
            second = pairs[:, 1]
            has_padding = self._padded_to != 0 or self._fixed_n_pairs != 0
            if has_padding:
                # Padding pseudo-pairs are `(i, -1)` or `(-1, -1)`, with zero shift.
                # Replacing them with `(0, 0)` gives them a zero vector without
                # indexing outside the points tensor.
                padding = second < 0
                first = first.masked_fill(padding, 0)
                second = second.masked_fill(padding, 0)
            vectors = (
                points_f64.index_select(0, second)
                - points_f64.index_select(0, first)
                + shifts.to(torch.float64) @ box_f64
            )
            if has_padding and return_distances:
                padding = pairs[:, 1] < 0
                # The second derivative of the norm is undefined at zero.
                safe_vectors = vectors.masked_fill(padding[:, None], 1.0)
                distances = torch.linalg.vector_norm(safe_vectors, dim=1).masked_fill(
                    padding, 0.0
                )
            else:
                distances = torch.linalg.vector_norm(vectors, dim=1)
        else:
            pairs, shifts, distances, vectors = torch.ops.vesin.neighbors(
                self._state,
                points_f64,
                box_f64,
                periodic,
                self._cutoff,
                self._full_list,
                self._sorted,
                self._padded_to,
                self._fixed_n_pairs,
                self._algorithm,
                self._skin,
                self._n_threads,
                return_shifts,
                return_distances,
                return_vectors,
                copy,
            )

        results: List[torch.Tensor] = []
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
                results.append(distances.to(initial_dtype))
            elif q == "D":
                results.append(vectors.to(initial_dtype))
        return results
