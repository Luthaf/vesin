import ctypes
import os
import sys
from ctypes import cdll

import jax


_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    windows = False
    if sys.platform.startswith("darwin"):
        path = os.path.join(_HERE, "libvesin_jax.dylib")
    elif sys.platform.startswith("linux"):
        path = os.path.join(_HERE, "libvesin_jax.so")
    elif sys.platform.startswith("win"):
        path = os.path.join(_HERE, "vesin_jax.dll")
        windows = True
    else:
        raise ImportError("Unknown platform. Please edit this file")

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path
    else:
        raise ImportError("Could not find vesin_jax shared library at " + path)


def _check_dll(path):
    """
    Check if the DLL pointer size matches Python (32-bit or 64-bit)
    """
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404

    machine = None
    with open(path, "rb") as fd:
        header = fd.read(2).decode(encoding="utf-8", errors="strict")
        if header != "MZ":
            raise ImportError(path + " is not a DLL")
        else:
            fd.seek(60)
            header = fd.read(4)
            header_offset = struct.unpack("<L", header)[0]
            fd.seek(header_offset + 4)
            header = fd.read(2)
            machine = struct.unpack("<H", header)[0]

    arch = platform.architecture()[0]
    if arch == "32bit":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit, but this DLL is not")
    elif arch == "64bit":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit, but this DLL is not")
    else:
        raise ImportError("Could not determine pointer size of Python")


_FFI_TARGETS = [
    "vesin_jax_compute",
    "vesin_jax_compute_dynamic",
    "vesin_jax_copy_out",
]


class LibraryFinder(object):
    def __init__(self):
        self._cached_dll = None

    def __call__(self):
        if self._cached_dll is None:
            path = _lib_path()
            self._cached_dll = cdll.LoadLibrary(path)
            _setup_functions(self._cached_dll)

        return self._cached_dll


def _setup_functions(lib):
    lib.vesin_jax_state_create.argtypes = []
    lib.vesin_jax_state_create.restype = ctypes.c_void_p

    lib.vesin_jax_state_free.argtypes = [ctypes.c_void_p]
    lib.vesin_jax_state_free.restype = None

    jax.ffi.register_ffi_target(
        "vesin_jax_compute",
        jax.ffi.pycapsule(lib.vesin_jax_compute),
        platform="cpu",
    )

    jax.ffi.register_ffi_target(
        "vesin_jax_compute_dynamic",
        jax.ffi.pycapsule(lib.vesin_jax_compute_dynamic),
        platform="cpu",
    )

    jax.ffi.register_ffi_target(
        "vesin_jax_copy_out",
        jax.ffi.pycapsule(lib.vesin_jax_copy_out),
        platform="cpu",
    )


get_library = LibraryFinder()
