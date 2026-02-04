import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


ROOT = os.path.realpath(os.path.dirname(__file__))

VESIN_BUILD_TYPE = os.environ.get("VESIN_BUILD_TYPE", "Release")
if VESIN_BUILD_TYPE not in ["Debug", "Release"]:
    raise Exception(
        f"invalid build type passed: '{VESIN_BUILD_TYPE}', "
        "expected 'Debug' or 'Release'"
    )


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.7. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """
    Build the native library using cmake
    """

    def run(self):
        source_dir = os.path.join(ROOT, "lib")
        if not os.path.exists(source_dir):
            # we are building from a checkout
            source_dir = os.path.join(ROOT, "..", "..", "vesin")

        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "vesin")

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={VESIN_BUILD_TYPE}",
            "-DBUILD_SHARED_LIBS=ON",
        ]

        CUDA_HOME = os.environ.get("CUDA_HOME")
        VESIN_ENABLE_NVTX = os.environ.get("VESIN_ENABLE_NVTX", "OFF")

        if CUDA_HOME is not None:
            cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}")
            cmake_options.append("-DVESIN_ENABLE_CUDA=ON")
            if VESIN_ENABLE_NVTX.upper() in ["ON", "1", "TRUE", "YES"]:
                cmake_options.append("-DVESIN_ENABLE_NVTX=ON")

            # fix for https://github.com/pytorch/pytorch/issues/113948, it does not
            # matter which architecture we put here since we are not actually compiling
            # any CUDA code, just linking against cudart
            cmake_options.append("-DTORCH_CUDA_ARCH_LIST=9.0")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )

        subprocess.run(
            [
                "cmake",
                "--build",
                build_dir,
                "--config",
                VESIN_BUILD_TYPE,
                "--target",
                "install",
            ],
            check=True,
        )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs.\nUse `pip install .` or "
            "`python -m build --wheel . && pip install "
            "dist/metatensor_core-*.whl` to install from source."
        )


class sdist_with_lib(sdist):
    """
    Create a sdist including the code for the native library
    """

    def run(self):
        # generate extra files
        shutil.copytree(os.path.join(ROOT, "..", "..", "vesin"), "lib")

        # run original sdist
        super().run()

        # cleanup
        shutil.rmtree("lib")


if __name__ == "__main__":
    setup(
        version=open("VERSION").read().strip(),
        ext_modules=[
            # only declare the extension, it is built & copied as required by cmake
            # in the build_ext command
            Extension(name="vesin", sources=[]),
        ],
        cmdclass={
            "sdist": sdist_with_lib,
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
        },
    )
