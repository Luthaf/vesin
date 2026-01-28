import ctypes
from ctypes import POINTER
from typing import List, Sequence, Union

import numpy as np
import numpy.typing as npt

from ._c_api import (
    VesinAutoAlgorithm,
    VesinBruteForce,
    VesinCellList,
    VesinCPU,
    VesinCUDA,
    VesinDevice,
    VesinNeighborList,
    VesinOptions,
    VesinUnknownDevice,
)
from ._c_lib import _get_library


try:
    import cupy as cp

    HAS_CUPY = True

    CUPY_DTYPES = {
        ctypes.c_size_t: cp.uint64,
        ctypes.c_int32: cp.int32,
        ctypes.c_double: cp.float64,
    }
except ImportError:
    HAS_CUPY = False
    cp = None


def _ptr_to_numpy(ptr, shape, dtype, owner, device_id):
    """
    Helper to convert a ctypes pointer to a NumPy array.
    """
    return np.ctypeslib.as_array(ctypes.cast(ptr, POINTER(dtype)), shape=shape)


def _ptr_to_cupy(ptr, shape, dtype, owner, device_id):
    """
    Helper to convert a ctypes pointer to a CuPy array.
    """
    ptr_val = ctypes.cast(ptr, ctypes.c_void_p).value
    size = np.prod(shape) * ctypes.sizeof(dtype)
    mem = cp.cuda.memory.UnownedMemory(ptr_val, size, owner, device_id)
    return cp.ndarray(
        shape, dtype=CUPY_DTYPES[dtype], memptr=cp.cuda.MemoryPointer(mem, 0)
    )


def _device_from_array(array):
    """
    Determine the VesinDevice from a numpy or cupy array.

    :param array: numpy array or cupy array
    :return: VesinDevice structure
    """
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return VesinDevice(VesinCUDA, array.device.id)
    else:
        return VesinDevice(VesinCPU, 0)


class NeighborList:
    """
    A neighbor list calculator.
    """

    def __init__(
        self,
        cutoff: float,
        full_list: bool,
        sorted: bool = False,
        algorithm: str = "auto",
    ):
        """
        :param cutoff: spherical cutoff for this neighbor list
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``) or
            only once
        :param sorted: Should vesin sort the returned pairs in lexicographic order
            (sorting both ``i`` and then ``j`` at constant ``i``)?
        :param algorithm: algorithm to use when computing the neighbor list. One of
            ``"auto"``, ``"brute_force"``, or ``"cell_list"``.
        """
        self._lib = _get_library()
        self.cutoff = float(cutoff)
        self.full_list = bool(full_list)
        self.sorted = bool(sorted)

        self.algorithm = algorithm

        self._neighbors = VesinNeighborList()

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_neighbors"):
            self._lib.vesin_free(self._neighbors)

    @property
    def algorithm(self) -> str:
        """Get the current algorithm used for NL computation."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: str):
        """Set the algorithm to use for NL computation."""
        algorithm = value.lower()
        self._algorithm = algorithm
        if algorithm == "auto":
            self._c_algorithm = VesinAutoAlgorithm
        elif algorithm == "brute_force":
            self._c_algorithm = VesinBruteForce
        elif algorithm == "cell_list":
            self._c_algorithm = VesinCellList
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                "Must be one of 'auto', 'brute_force', or 'cell_list'."
            )

    def compute(
        self,
        points: "npt.ArrayLike",
        box: "npt.ArrayLike",
        periodic: Union[bool, Sequence[bool], "npt.ArrayLike"],
        quantities: str = "ij",
        copy=True,
    ) -> List[np.ndarray]:
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

        :param points: positions of all points in the system (this can be anything that
            can be converted to a numpy or cupy array)
        :param box: bounding box of the system (this can be anything that can be
            converted to a numpy or cupy array)
        :param periodic: should we use periodic boundary conditions? This can be a
            single boolean to enable/disable periodic boundary conditions in all
            directions, or a sequence of three booleans (one for each direction).
        :param quantities: quantities to return, defaults to "ij"
        :param copy: should we copy the returned quantities, defaults to ``True``.
            Setting this to ``False`` might be a bit faster, but the returned arrays are
            view inside this class, and will be invalidated whenever this class is
            garbage collected or used to run a new calculation.

        :return: tuple of arrays as indicated by ``quantities``.
        """
        # Detect if input is CuPy array and convert to numpy for CPU processing
        # The C library will handle GPU computation internally when device is CUDA
        is_cupy = HAS_CUPY and isinstance(points, cp.ndarray)

        if box.shape != (3, 3):
            raise ValueError("`box` must be a 3x3 matrix")

        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError("`points` must be a nx3 array")

        options = VesinOptions()
        options.cutoff = self.cutoff
        options.full = self.full_list
        options.sorted = self.sorted
        options.return_shifts = "S" in quantities
        options.return_distances = "d" in quantities
        options.return_vectors = "D" in quantities
        options.algorithm = self._c_algorithm

        if isinstance(periodic, (bool, np.bool_)):
            periodic = np.array([periodic, periodic, periodic], dtype=np.bool_)

        if is_cupy:
            periodic = cp.asarray(periodic, dtype=cp.bool_)
        else:
            periodic = np.asarray(periodic, dtype=np.bool_)

        if periodic.shape != (3,):
            raise ValueError(
                "`periodic` must be a single boolean or a sequence of three "
                f"booleans, got {periodic} of type {type(periodic)}"
            )

        # Get device and data pointer
        device = _device_from_array(points)

        if is_cupy:
            points = cp.asarray(points, dtype=cp.float64)
            box = cp.asarray(box, dtype=cp.float64)
            # Ensure C-contiguous
            if not points.flags.c_contiguous:
                points = cp.ascontiguousarray(points)
            if not box.flags.c_contiguous:
                box = cp.ascontiguousarray(box)
            if not periodic.flags.c_contiguous:
                periodic = cp.ascontiguousarray(periodic)

            points_ptr = ctypes.cast(
                ctypes.c_void_p(points.data.ptr), POINTER(ctypes.c_double)
            )
            box_ptr = ctypes.cast(
                ctypes.c_void_p(box.data.ptr), POINTER(ctypes.c_double)
            )
            periodic_ptr = ctypes.cast(
                ctypes.c_void_p(periodic.data.ptr), POINTER(ctypes.c_bool)
            )
        else:
            points = np.asarray(points, dtype=np.float64)
            box = np.asarray(box, dtype=np.float64)
            # Ensure C-contiguous
            if not points.flags.c_contiguous:
                points = np.ascontiguousarray(points)
            if not box.flags.c_contiguous:
                box = np.ascontiguousarray(box)
            if not periodic.flags.c_contiguous:
                periodic = np.ascontiguousarray(periodic)

            points_ptr = points.ctypes.data_as(POINTER(ctypes.c_double))
            box_ptr = box.ctypes.data_as(POINTER(ctypes.c_double))
            periodic_ptr = periodic.ctypes.data_as(POINTER(ctypes.c_bool))

        if self._neighbors.device.type != VesinUnknownDevice:
            if (
                self._neighbors.device.type != device.type
                or self._neighbors.device.device_id != device.device_id
            ):
                # Free previous allocation and reset to zeroed state
                self._lib.vesin_free(self._neighbors)
                ctypes.memset(
                    ctypes.byref(self._neighbors),
                    0,
                    ctypes.sizeof(self._neighbors),
                )

        error_message = ctypes.c_char_p()
        status = self._lib.vesin_neighbors(
            points_ptr,
            points.shape[0],
            box_ptr,
            periodic_ptr,
            device,
            options,
            ctypes.byref(self._neighbors),
            error_message,
        )

        if status != 0:
            raise RuntimeError(error_message.value.decode("utf8"))

        # Create arrays for the output data
        n_pairs = self._neighbors.length

        if is_cupy:
            _empty_array = cp.empty
            _ptr_to_array = _ptr_to_cupy
        else:
            _empty_array = np.empty
            _ptr_to_array = _ptr_to_numpy

        if n_pairs == 0:
            # Empty results
            pairs = _empty_array((0, 2), dtype=ctypes.c_size_t)
            shifts = _empty_array((0, 3), dtype=ctypes.c_int32)
            distances = _empty_array((0,), dtype=ctypes.c_double)
            vectors = _empty_array((0, 3), dtype=ctypes.c_double)
        else:
            pairs = _ptr_to_array(
                self._neighbors.pairs,
                shape=(n_pairs, 2),
                dtype=ctypes.c_size_t,
                owner=self._neighbors,
                device_id=self._neighbors.device.device_id,
            )
            if "S" in quantities:
                shifts = _ptr_to_array(
                    self._neighbors.shifts,
                    shape=(n_pairs, 3),
                    dtype=ctypes.c_int32,
                    owner=self._neighbors,
                    device_id=self._neighbors.device.device_id,
                )
            if "d" in quantities:
                distances = _ptr_to_array(
                    self._neighbors.distances,
                    shape=(n_pairs,),
                    dtype=ctypes.c_double,
                    owner=self._neighbors,
                    device_id=self._neighbors.device.device_id,
                )
            if "D" in quantities:
                vectors = _ptr_to_array(
                    self._neighbors.vectors,
                    shape=(n_pairs, 3),
                    dtype=ctypes.c_double,
                    owner=self._neighbors,
                    device_id=self._neighbors.device.device_id,
                )

        # assemble output

        data = []
        for quantity in quantities:
            if quantity == "P":
                if copy:
                    data.append(pairs.copy())
                else:
                    data.append(pairs)

            if quantity == "i":
                if copy:
                    data.append(pairs[:, 0].copy())
                else:
                    data.append(pairs[:, 0])

            elif quantity == "j":
                if copy:
                    data.append(pairs[:, 1].copy())
                else:
                    data.append(pairs[:, 1])

            elif quantity == "S":
                if copy:
                    data.append(shifts.copy())
                else:
                    data.append(shifts)

            elif quantity == "d":
                if copy:
                    data.append(distances.copy())
                else:
                    data.append(distances)

            elif quantity == "D":
                if copy:
                    data.append(vectors.copy())
                else:
                    data.append(vectors)

        return tuple(data)
