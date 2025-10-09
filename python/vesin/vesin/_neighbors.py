import ctypes
from ctypes import ARRAY, POINTER
from typing import List

import numpy as np
import numpy.typing as npt

from ._c_api import (
    HAS_CUPY,
    VesinNeighborList,
    VesinOptions,
    get_device_from_array,
)
from ._c_lib import _get_library


if HAS_CUPY:
    import cupy as cp


class NeighborList:
    """
    A neighbor list calculator.
    """

    def __init__(self, cutoff: float, full_list: bool, sorted: bool = False):
        """
        :param cutoff: spherical cutoff for this neighbor list
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``) or
            only once
        :param sorted: Should vesin sort the returned pairs in lexicographic order
            (sorting both ``i`` and then ``j`` at constant ``i``)?
        """
        self._lib = _get_library()
        self.cutoff = float(cutoff)
        self.full_list = bool(full_list)
        self.sorted = bool(sorted)

        self._neighbors = VesinNeighborList()

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_neighbors"):
            self._lib.vesin_free(self._neighbors)

    def compute(
        self,
        points: "npt.ArrayLike",
        box: "npt.ArrayLike",
        periodic: bool,
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
        :param periodic: should we use periodic boundary conditions?
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

        if is_cupy:
            points_gpu = cp.asarray(points, dtype=cp.float64)
            box_gpu = cp.asarray(box, dtype=cp.float64)
            # Ensure C-contiguous
            if not points_gpu.flags.c_contiguous:
                points_gpu = cp.ascontiguousarray(points_gpu)
            if not box_gpu.flags.c_contiguous:
                box_gpu = cp.ascontiguousarray(box_gpu)
            points = points_gpu
            box = box_gpu
        else:
            points = np.asarray(points, dtype=np.float64)
            box = np.asarray(box, dtype=np.float64)
            # Ensure C-contiguous
            if not points.flags.c_contiguous:
                points = np.ascontiguousarray(points)
            if not box.flags.c_contiguous:
                box = np.ascontiguousarray(box)

        if box.shape != (3, 3):
            raise ValueError("`box` must be a 3x3 matrix")

        # Prepare box pointer
        Vector = ARRAY(ctypes.c_double, 3)
        BoxArray = ARRAY(Vector, 3)
        if is_cupy:
            # For CUDA: pass raw GPU pointer
            box_ptr = ctypes.c_void_p(box.data.ptr)
        else:
            # For CPU: convert to ctypes structure
            box_ptr = BoxArray(
                Vector(box[0][0], box[0][1], box[0][2]),
                Vector(box[1][0], box[1][1], box[1][2]),
                Vector(box[2][0], box[2][1], box[2][2]),
            )

        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError("`points` must be a nx3 array")

        options = VesinOptions()
        options.cutoff = self.cutoff
        options.full = self.full_list
        options.sorted = self.sorted
        options.return_shifts = "S" in quantities
        options.return_distances = "d" in quantities
        options.return_vectors = "D" in quantities

        # Get device and data pointer
        device = get_device_from_array(points)

        if is_cupy:
            # For CUDA: pass raw GPU pointer
            points_ptr = ctypes.c_void_p(points.data.ptr)
        else:
            # For CPU: use numpy's ctypes interface
            points_ptr = points.ctypes.data_as(POINTER(ARRAY(ctypes.c_double, 3)))

        error_message = ctypes.c_char_p()

        # For CUDA, we need to bypass ctypes type checking since the pointers are on GPU
        if is_cupy:
            # Temporarily set argtypes to None to bypass type checking
            original_argtypes = self._lib.vesin_neighbors.argtypes
            self._lib.vesin_neighbors.argtypes = None
            status = self._lib.vesin_neighbors(
                points_ptr,
                points.shape[0],
                box_ptr,
                periodic,
                device,
                options,
                ctypes.byref(self._neighbors),
                ctypes.byref(error_message),
            )
            self._lib.vesin_neighbors.argtypes = original_argtypes
        else:
            status = self._lib.vesin_neighbors(
                points_ptr,
                points.shape[0],
                box_ptr,
                periodic,
                device,
                options,
                self._neighbors,
                error_message,
            )

        if status != 0:
            if error_message.value:
                raise RuntimeError(error_message.value.decode("utf8"))
            else:
                raise RuntimeError(
                    f"vesin_neighbors failed with status {status} but no error message"
                )

        # Create arrays for the output data
        n_pairs = self._neighbors.length

        if n_pairs == 0:
            # Empty results
            array_mod = cp if is_cupy else np
            pairs = array_mod.empty((0, 2), dtype=ctypes.c_size_t)
            shifts = array_mod.empty((0, 3), dtype=ctypes.c_int32)
            distances = array_mod.empty((0,), dtype=ctypes.c_double)
            vectors = array_mod.empty((0, 3), dtype=ctypes.c_double)
        elif is_cupy:
            # CUDA results - wrap GPU memory as CuPy arrays
            device_id = self._neighbors.device.device_id

            # Helper to create CuPy array from GPU pointer
            def wrap_gpu_array(ptr, shape, dtype_cp, dtype_c):
                ptr_val = ctypes.cast(ptr, ctypes.c_void_p).value
                size = np.prod(shape) * ctypes.sizeof(dtype_c)
                mem = cp.cuda.memory.UnownedMemory(
                    ptr_val, size, self._neighbors, device_id
                )
                return cp.ndarray(
                    shape, dtype=dtype_cp, memptr=cp.cuda.MemoryPointer(mem, 0)
                )

            pairs = wrap_gpu_array(
                self._neighbors.pairs, (n_pairs, 2), cp.uint64, ctypes.c_size_t
            )
            if "S" in quantities:
                shifts = wrap_gpu_array(
                    self._neighbors.shifts, (n_pairs, 3), cp.int32, ctypes.c_int32
                )
            if "d" in quantities:
                distances = wrap_gpu_array(
                    self._neighbors.distances, (n_pairs,), cp.float64, ctypes.c_double
                )
            if "D" in quantities:
                vectors = wrap_gpu_array(
                    self._neighbors.vectors, (n_pairs, 3), cp.float64, ctypes.c_double
                )
        else:
            # CPU results - wrap as NumPy arrays
            pairs = np.ctypeslib.as_array(
                ctypes.cast(self._neighbors.pairs, POINTER(ctypes.c_size_t)),
                shape=(n_pairs, 2),
            )
            if "S" in quantities:
                shifts = np.ctypeslib.as_array(
                    ctypes.cast(self._neighbors.shifts, POINTER(ctypes.c_int32)),
                    shape=(n_pairs, 3),
                )
            if "d" in quantities:
                distances = np.ctypeslib.as_array(
                    ctypes.cast(self._neighbors.distances, POINTER(ctypes.c_double)),
                    shape=(n_pairs,),
                )
            if "D" in quantities:
                vectors = np.ctypeslib.as_array(
                    ctypes.cast(self._neighbors.vectors, POINTER(ctypes.c_double)),
                    shape=(n_pairs, 3),
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
