from typing import Any, Dict, List, Union

import torch


class NeighborList:
    """A neighbor list calculator that can be used with TorchScript."""

    def __init__(
        self,
        cutoff: float,
        full_list: bool,
        sorted: bool = False,
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
                self._algorithm,
                self._skin,
                self._n_threads,
                True,
                False,
                False,
                True,
            )

            vectors = (
                points_f64.index_select(0, pairs[:, 1])
                - points_f64.index_select(0, pairs[:, 0])
                + shifts.to(torch.float64) @ box_f64
            )
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
