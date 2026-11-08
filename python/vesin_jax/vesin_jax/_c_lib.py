import ctypes
import os
import sys

import jax


_HERE = os.path.realpath(os.path.dirname(__file__))

# name of the FFI target used by the `vesin_neighbors` primitive
FFI_TARGET = "vesin_neighbors"

_LIB = None


def _lib_path():
    if sys.platform.startswith("darwin"):
        path = os.path.join(_HERE, "lib", "libvesin_jax.dylib")
    elif sys.platform.startswith("linux"):
        path = os.path.join(_HERE, "lib", "libvesin_jax.so")
    elif sys.platform.startswith("win"):
        path = os.path.join(_HERE, "bin", "vesin_jax.dll")
    else:
        raise ImportError("Unknown platform. Please edit this file")

    if not os.path.isfile(path):
        raise ImportError(f"Could not find vesin_jax shared library at {path}")

    return path


def register_ffi_targets():
    """
    Load the native library and register its handlers as XLA FFI targets for
    CPU and CUDA.
    """
    global _LIB
    lib = ctypes.cdll.LoadLibrary(_lib_path())
    lib.vesin_jax_free_state.argtypes = [ctypes.c_int64]
    lib.vesin_jax_free_state.restype = None
    _LIB = lib

    jax.ffi.register_ffi_target(
        FFI_TARGET,
        jax.ffi.pycapsule(lib.vesin_jax_neighbors_cpu),
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        FFI_TARGET,
        jax.ffi.pycapsule(lib.vesin_jax_neighbors_cuda),
        platform="CUDA",
    )


def free_state(state):
    """Release the memory used by the state of a ``NeighborList``"""
    _LIB.vesin_jax_free_state(state)
