import glob
import os
import re
import shutil
import subprocess
import sys
import urllib.request

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
        import torch

        source_dir = os.path.join(ROOT, "lib")
        if not os.path.exists(source_dir):
            # we are building from a checkout
            source_dir = os.path.join(ROOT, "..", "..", "torch")

        build_dir = os.path.join(ROOT, "build", "cmake-build")

        # Install the shared library in a prefix matching the torch version used to
        # compile the code. This allows having multiple version of this shared library
        # inside the wheel; and dynamically pick the right one.
        torch_major, torch_minor, *_ = torch.__version__.split(".")
        install_dir = os.path.join(
            os.path.realpath(self.build_lib),
            "vesin",
            "torch",
            f"torch-{torch_major}.{torch_minor}",
        )

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={VESIN_BUILD_TYPE}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
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

        # do not include the non-torch vesin lib in the wheel
        for file in glob.glob(os.path.join(install_dir, "bin", "*")):
            if "vesin_torch" not in os.path.basename(file):
                os.unlink(file)

        for file in glob.glob(os.path.join(install_dir, "lib", "*")):
            if "vesin_torch" not in os.path.basename(file):
                os.unlink(file)

        for file in glob.glob(os.path.join(install_dir, "include", "*")):
            if "vesin_torch" not in os.path.basename(file):
                os.unlink(file)


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
        shutil.copytree(os.path.join(ROOT, "..", "..", "torch"), os.path.join("lib"))

        # include gpulite in the sdist
        gpulite_dir = os.path.join("lib", "vesin", "external")
        os.makedirs(gpulite_dir, exist_ok=True)
        gpulite_archive = os.path.join(gpulite_dir, "gpulite.tar.gz")
        assert not os.path.exists(gpulite_archive)

        with open(os.path.join("lib", "vesin", "CMakeLists.txt")) as fd:
            content = fd.read()
            # FetchContent_Declare(
            #     gpulite
            #     ...
            #     GIT_TAG <hash>
            match = re.search(
                r"FetchContent_Declare\s*\(\s*gpulite.*?GIT_TAG\s+([a-f0-9]+)",
                content,
                re.DOTALL,
            )
            if match is None:
                raise Exception("Could not find gpulite GIT_TAG in CMakeLists.txt")
            commit = match.group(1)

        print("downloading gpulite source code")
        urllib.request.urlretrieve(
            f"https://github.com/rubber-duck-debug/gpu-lite/archive/{commit}.tar.gz",
            gpulite_archive,
        )

        # run original sdist
        super().run()

        # cleanup
        shutil.rmtree("lib")


if __name__ == "__main__":
    if sys.platform == "win32":
        # On Windows, starting with PyTorch 2.3, the file shm.dll in torch has a
        # dependency on mkl DLLs. When building the code using pip build isolation, pip
        # installs the mkl package in a place where the os is not trying to load
        #
        # This is a very similar fix to https://github.com/pytorch/pytorch/pull/126095,
        # except only applying when importing torch from a build-isolation virtual
        # environment created by pip (`python -m build` does not seems to suffer from
        # this).
        import wheel

        pip_virtualenv = os.path.realpath(
            os.path.join(
                os.path.dirname(wheel.__file__),
                "..",
                "..",
                "..",
                "..",
            )
        )
        mkl_dll_dir = os.path.join(
            pip_virtualenv,
            "normal",
            "Library",
            "bin",
        )

        if os.path.exists(mkl_dll_dir):
            os.add_dll_directory(mkl_dll_dir)

        # End of Windows/MKL/PIP hack

    try:
        import torch

        # if we have torch, we are building a wheel, which will only be compatible with
        # a single torch version
        torch_v_major, torch_v_minor, *_ = torch.__version__.split(".")
        torch_version = f"== {torch_v_major}.{torch_v_minor}.*"
    except ImportError:
        # otherwise we are building a sdist
        torch_version = ">= 2.1"

    install_requires = [f"torch {torch_version}"]

    setup(
        version=open("VERSION").read().strip(),
        install_requires=install_requires,
        ext_modules=[
            # only declare the extension, it is built & copied as required by cmake
            # in the build_ext command
            Extension(name="vesin_torch", sources=[]),
        ],
        cmdclass={
            "sdist": sdist_with_lib,
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
        },
    )
