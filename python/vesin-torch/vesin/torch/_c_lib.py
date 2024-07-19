import glob
import os
import re
import sys
from collections import namedtuple

import torch

Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():

    torch_version = parse_version(torch.__version__)
    expected_prefix = os.path.join(
        _HERE, f"torch-{torch_version.major}.{torch_version.minor}"
    )
    if os.path.exists(expected_prefix):
        if sys.platform.startswith("darwin"):
            path = os.path.join(expected_prefix, "lib", "libvesin_torch.dylib")
            windows = False
        elif sys.platform.startswith("linux"):
            path = os.path.join(expected_prefix, "lib", "libvesin_torch.so")
            windows = False
        elif sys.platform.startswith("win"):
            path = os.path.join(expected_prefix, "bin", "vesin_torch.dll")
            windows = True
        else:
            raise ImportError("Unknown platform. Please edit this file")

        if os.path.isfile(path):
            if windows:
                _check_dll(path)
            return path
        else:
            raise ImportError("Could not find vesin_torch shared library at " + path)

    # gather which torch version(s) the current install was built
    # with to create the error message
    existing_versions = []
    for prefix in glob.glob(os.path.join(_HERE, "torch-*")):
        existing_versions.append(os.path.basename(prefix)[6:])

    if len(existing_versions) == 1:
        raise ImportError(
            f"Trying to load vesin-torch with torch v{torch.__version__}, "
            f"but it was compiled against torch v{existing_versions[0]}, which "
            "is not ABI compatible"
        )
    else:
        all_versions = ", ".join(map(lambda version: f"v{version}", existing_versions))
        raise ImportError(
            f"Trying to load vesin-torch with torch v{torch.__version__}, "
            f"we found builds for torch {all_versions}; which are not ABI compatible.\n"
            "You can try to re-install from source with "
            "`pip install vesin-torch --no-binary=vesin-torch`"
        )


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


def _load_library():
    # load the C++ operators and custom classes
    try:
        torch.ops.load_library(_lib_path())
    except Exception as e:
        if "undefined symbol" in str(e):
            file_name = os.path.basename(_lib_path())
            raise ImportError(
                f"{file_name} is not compatible with the current PyTorch "
                "installation.\nThis can happen if PyTorch comes from one source "
                "(pip, conda, custom), but vesin-torch comes from a different "
                "one.\nIn this case, you can try to compile vesin-torch yourself "
                "with `pip install vesin-torch --no-binary=vesin-torch`"
            ) from e
        else:
            raise e
