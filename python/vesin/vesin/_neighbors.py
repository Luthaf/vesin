import ctypes
import glob
import os
import sys
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

try:
    import torch

    HAS_TORCH = True

    TORCH_DTYPES = {
        ctypes.c_size_t: torch.uint64,
        ctypes.c_int32: torch.int32,
        ctypes.c_double: torch.float64,
    }
except ImportError:
    HAS_TORCH = False


def _device_from_array(array):
    """
    Determine the VesinDevice from a numpy or cupy array.

    :param array: numpy array or cupy array
    :return: VesinDevice structure
    """
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return VesinDevice(VesinCUDA, array.device.id)
    elif HAS_TORCH and isinstance(array, torch.Tensor):
        if array.device.type == "cuda":
            return VesinDevice(VesinCUDA, array.device.index or 0)
        else:
            return VesinDevice(VesinCPU, 0)
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
        use_cupy = HAS_CUPY and isinstance(points, cp.ndarray)
        use_torch = HAS_TORCH and isinstance(points, torch.Tensor)

        if use_cupy:
            get_ptr_fn = _cupy_get_ptr
            as_array_fn = _cupy_asarray
            to_dtype_fn = _cupy_to_dtype
            float64_dtype = cp.float64
            bool_dtype = cp.bool_
            empty_array_fn = _cupy_empty
            ptr_to_array_fn = _ptr_to_cupy
            copy_array_fn = lambda arr: arr.copy()  # noqa: E731
        elif use_torch:
            get_ptr_fn = _torch_get_ptr
            as_array_fn = _torch_asarray
            to_dtype_fn = _torch_to_dtype
            float64_dtype = torch.float64
            bool_dtype = torch.bool
            empty_array_fn = _torch_empty
            ptr_to_array_fn = _ptr_to_torch
            copy_array_fn = lambda arr: arr.clone()  # noqa: E731
        else:
            # use numpy by default
            get_ptr_fn = _numpy_get_ptr
            as_array_fn = _numpy_asarray
            to_dtype_fn = _numpy_to_dtype
            float64_dtype = np.float64
            bool_dtype = np.bool_
            empty_array_fn = _numpy_empty
            ptr_to_array_fn = _ptr_to_numpy
            copy_array_fn = lambda arr: arr.copy()  # noqa: E731

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

        # Get device and data pointer
        device = _device_from_array(points)

        points = as_array_fn(points, device_like=None)
        box = as_array_fn(box, device_like=None)
        periodic = as_array_fn(periodic, device_like=points)
        periodic = to_dtype_fn(periodic, bool_dtype)

        if periodic.shape != (3,):
            raise ValueError(
                "`periodic` must be a single boolean or a sequence of three "
                f"booleans, got {periodic} of type {type(periodic)}"
            )

        initial_dtype = points.dtype
        if box.dtype != initial_dtype:
            raise RuntimeError(
                "`points` and `box` must have the same dtype, "
                f"got {points.dtype} and {box.dtype}"
            )

        points = to_dtype_fn(points, float64_dtype)
        box = to_dtype_fn(box, float64_dtype)

        if use_torch:
            if points.requires_grad or box.requires_grad:
                raise RuntimeError(
                    "NeighborList.compute does not support autograd for torch tensors"
                )

        points, points_ptr = get_ptr_fn(points, ctypes.c_double)
        box, box_ptr = get_ptr_fn(box, ctypes.c_double)
        periodic, periodic_ptr = get_ptr_fn(periodic, ctypes.c_bool)

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

        if n_pairs == 0:
            # Empty results -- use initial_dtype for floating-point arrays so
            # the caller gets the same dtype they passed in.
            pairs = empty_array_fn((0, 2), dtype=ctypes.c_size_t, device_like=points)
            shifts = empty_array_fn((0, 3), dtype=ctypes.c_int32, device_like=points)
            distances = empty_array_fn((0,), dtype=initial_dtype, device_like=points)
            vectors = empty_array_fn((0, 3), dtype=initial_dtype, device_like=points)
        else:
            pairs = ptr_to_array_fn(
                self._neighbors.pairs,
                shape=(n_pairs, 2),
                dtype=ctypes.c_size_t,
                owner=self._neighbors,
                device=self._neighbors.device,
            )
            if "S" in quantities:
                shifts = ptr_to_array_fn(
                    self._neighbors.shifts,
                    shape=(n_pairs, 3),
                    dtype=ctypes.c_int32,
                    owner=self._neighbors,
                    device=self._neighbors.device,
                )
            if "d" in quantities:
                distances = ptr_to_array_fn(
                    self._neighbors.distances,
                    shape=(n_pairs,),
                    dtype=ctypes.c_double,
                    owner=self._neighbors,
                    device=self._neighbors.device,
                )
                distances = to_dtype_fn(distances, initial_dtype)
            if "D" in quantities:
                vectors = ptr_to_array_fn(
                    self._neighbors.vectors,
                    shape=(n_pairs, 3),
                    dtype=ctypes.c_double,
                    owner=self._neighbors,
                    device=self._neighbors.device,
                )
                vectors = to_dtype_fn(vectors, initial_dtype)

        # assemble output

        data = []
        for quantity in quantities:
            if quantity == "P":
                if copy:
                    data.append(copy_array_fn(pairs))
                else:
                    data.append(pairs)

            if quantity == "i":
                if copy:
                    data.append(copy_array_fn(pairs[:, 0]))
                else:
                    data.append(pairs[:, 0])

            elif quantity == "j":
                if copy:
                    data.append(copy_array_fn(pairs[:, 1]))
                else:
                    data.append(pairs[:, 1])

            elif quantity == "S":
                if copy:
                    data.append(copy_array_fn(shifts))
                else:
                    data.append(shifts)

            elif quantity == "d":
                if copy:
                    data.append(copy_array_fn(distances))
                else:
                    data.append(distances)

            elif quantity == "D":
                if copy:
                    data.append(copy_array_fn(vectors))
                else:
                    data.append(vectors)

        return tuple(data)


########################################################################################


def _numpy_asarray(array, device_like) -> np.ndarray:
    return np.asarray(array)


def _numpy_to_dtype(array: np.ndarray, dtype) -> np.ndarray:
    return array.astype(dtype)


def _numpy_empty(shape, dtype, device_like) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def _numpy_get_ptr(array: np.ndarray, dtype) -> (np.ndarray, POINTER):
    array = np.ascontiguousarray(array)
    return array, array.ctypes.data_as(POINTER(dtype))


def _ptr_to_numpy(ptr, shape, dtype, owner, device):
    """
    Helper to convert a ctypes pointer to a NumPy array.
    """
    assert device.type == VesinCPU, "Array is not on CPU"
    return np.ctypeslib.as_array(ctypes.cast(ptr, POINTER(dtype)), shape=shape)


########################################################################################


def _cupy_asarray(array, device_like) -> "cp.ndarray":
    return cp.asarray(array)


def _cupy_to_dtype(array: "cp.ndarray", dtype) -> "cp.ndarray":
    return array.astype(dtype)


def _cupy_empty(shape, dtype, device_like) -> "cp.ndarray":
    return cp.empty(shape, dtype=dtype)


def _cupy_get_ptr(array: "cp.ndarray", dtype) -> ("cp.ndarray", POINTER):
    array = cp.ascontiguousarray(array)
    ptr = ctypes.cast(ctypes.c_void_p(array.data.ptr), POINTER(dtype))
    return array, ptr


def _ptr_to_cupy(ptr, shape, dtype, owner, device):
    """
    Helper to convert a ctypes pointer to a CuPy array.
    """
    assert device.type == VesinCUDA, "Array is not on CUDA device"
    ptr_val = ctypes.cast(ptr, ctypes.c_void_p).value
    size = np.prod(shape) * ctypes.sizeof(dtype)
    mem = cp.cuda.memory.UnownedMemory(ptr_val, size, owner, device.device_id)
    return cp.ndarray(
        shape, dtype=CUPY_DTYPES[dtype], memptr=cp.cuda.MemoryPointer(mem, 0)
    )


########################################################################################


def _torch_asarray(array, device_like) -> "torch.Tensor":
    tensor = torch.as_tensor(array)
    if device_like is not None:
        return tensor.to(device=device_like.device)
    else:
        return tensor


def _torch_to_dtype(array: "torch.Tensor", dtype) -> "torch.Tensor":
    return array.to(dtype)


def _torch_empty(shape, dtype, device_like) -> "torch.Tensor":
    dtype = TORCH_DTYPES.get(dtype, dtype)
    return torch.empty(size=shape, dtype=dtype, device=device_like.device)


def _torch_get_ptr(array: "torch.Tensor", dtype) -> ("torch.Tensor", POINTER):
    array = array.contiguous()
    return array, ctypes.cast(ctypes.c_void_p(array.data_ptr()), POINTER(dtype))


CUDART = None


def _get_cudart():
    global CUDART
    if CUDART is not None:
        return CUDART

    import ctypes.util

    # try ctypes.util.find_library first
    cudart_path = ctypes.util.find_library("cudart")
    candidates = [cudart_path] if cudart_path else []

    # platform-specific fallbacks
    if sys.platform.startswith("win"):
        # common env vars and locations on Windows
        cuda_paths = []
        if "CUDA_PATH" in os.environ:
            cuda_paths.append(os.environ["CUDA_PATH"])
        if "CUDA_HOME" in os.environ:
            cuda_paths.append(os.environ["CUDA_HOME"])
        # look for typical cudart DLL names under bin/lib
        for cp in cuda_paths:
            candidates += glob.glob(os.path.join(cp, "bin", "cudart64_*.dll"))
            candidates += glob.glob(os.path.join(cp, "lib", "x64", "cudart64_*.dll"))
        # search directories on PATH
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidates += glob.glob(os.path.join(p, "cudart64_*.dll"))
    else:
        # linux: common install locations
        candidates += glob.glob("/usr/local/cuda*/lib64/libcudart.so*")
        candidates += glob.glob("/usr/lib*/cuda*/lib64/libcudart.so*")

        # look through LD_LIBRARY_PATH
        for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            candidates += glob.glob(os.path.join(p, "libcudart.so*"))

    # unique, non-empty
    candidates = [c for c in set(candidates) if c and os.path.isfile(c)]

    last_err = None
    for path in candidates:
        try:
            CUDART = ctypes.CDLL(path)
            break
        except Exception as e:
            last_err = e
            continue

    if CUDART is None:
        raise RuntimeError(
            "Could not find cudart library, make sure it is available or "
            "install cupy. On Linux, libcudart.so.* should be in a standard location "
            "or in $LD_LIBRARY_PATH. On Windows %CUDA_PATH% should be set or "
            "cudart64_*.dll should be in the %PATH%."
        ) from last_err

    CUDART.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    CUDART.cudaMemcpy.restype = ctypes.c_int
    return CUDART


def _ptr_to_torch(ptr, shape, dtype, owner, device):
    """
    Helper to convert a ctypes pointer to a Torch tensor.
    """

    if device.type == VesinCPU:
        array = _ptr_to_numpy(ptr, shape, dtype, owner, device)
        return torch.from_numpy(array)
    elif device.type == VesinCUDA:
        if HAS_CUPY:
            cupy_array = _ptr_to_cupy(ptr, shape, dtype, owner, device)
            return torch.as_tensor(cupy_array)
        else:
            cudart = _get_cudart()

            # TODO: we could try using from_dlpack to avoid a copy
            device = torch.device(f"cuda:{device.device_id}")
            tensor = torch.empty(size=shape, dtype=TORCH_DTYPES[dtype], device=device)
            tensor_ptr = ctypes.c_void_p(tensor.data_ptr())
            status = cudart.cudaMemcpy(
                tensor_ptr,
                ptr,
                tensor.nbytes,
                2,  # cudaMemcpyDeviceToDevice
            )
            if status != 0:
                raise RuntimeError(
                    f"cudaMemcpy failed when creating torch tensor (code {status})"
                )
            return tensor
    else:
        raise RuntimeError("Unknown device for torch tensor")
