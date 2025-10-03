import ctypes
from ctypes import ARRAY, POINTER

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


VesinDeviceKind = ctypes.c_int
VesinUnknownDevice = 0
VesinCPU = 1
VesinCUDA = 2


class VesinDevice(ctypes.Structure):
    _fields_ = [
        ("type", VesinDeviceKind),
        ("device_id", ctypes.c_int),
    ]


class VesinOptions(ctypes.Structure):
    _fields_ = [
        ("cutoff", ctypes.c_double),
        ("full", ctypes.c_bool),
        ("sorted", ctypes.c_bool),
        ("return_shifts", ctypes.c_bool),
        ("return_distances", ctypes.c_bool),
        ("return_vectors", ctypes.c_bool),
    ]


class VesinNeighborList(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_size_t),
        ("device", VesinDevice),
        ("pairs", POINTER(ARRAY(ctypes.c_size_t, 2))),
        ("shifts", POINTER(ARRAY(ctypes.c_int32, 3))),
        ("distances", POINTER(ctypes.c_double)),
        ("vectors", POINTER(ARRAY(ctypes.c_double, 3))),
        ("opaque", ctypes.c_void_p),
    ]


def get_device_from_array(array):
    """
    Determine the VesinDevice from a numpy or cupy array.

    :param array: numpy array or cupy array
    :return: VesinDevice structure
    """
    if HAS_CUPY and isinstance(array, cp.ndarray):
        device_id = array.device.id
        return VesinDevice(VesinCUDA, device_id)
    else:
        return VesinDevice(VesinCPU, 0)


def setup_functions(lib):
    lib.vesin_free.argtypes = [POINTER(VesinNeighborList)]
    lib.vesin_free.restype = None

    # Note: In C, `const double box[3][3]` is actually a pointer when passed as a parameter,
    # but ctypes treats it as an array value for type checking. We keep the array type
    # for the signature but can pass either an array value or a pointer that looks like an array.
    lib.vesin_neighbors.argtypes = [
        POINTER(ARRAY(ctypes.c_double, 3)),  # points
        ctypes.c_size_t,  # n_points
        ARRAY(ARRAY(ctypes.c_double, 3), 3),  # box
        ctypes.c_bool,  # periodic
        VesinDevice,  # device
        VesinOptions,  # options
        POINTER(VesinNeighborList),  # neighbors
        POINTER(ctypes.c_char_p),  # error_message
    ]
    lib.vesin_neighbors.restype = ctypes.c_int
