import ctypes
from ctypes import ARRAY, POINTER


VesinDevice = ctypes.c_int
VesinUnknownDevice = 0
VesinCPU = 1


class VesinOptions(ctypes.Structure):
    _fields_ = [
        ("cutoff", ctypes.c_double),
        ("full", ctypes.c_bool),
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
    ]


def setup_functions(lib):
    lib.vesin_free.argtypes = [POINTER(VesinNeighborList)]
    lib.vesin_free.restype = None

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
