import glob
import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel


ROOT = os.path.realpath(os.path.dirname(__file__))

VESIN_BUILD_TYPE = os.environ.get("VESIN_BUILD_TYPE", "release")
if VESIN_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{VESIN_BUILD_TYPE}', "
        "expected 'debug' or 'release'"
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
            source_dir = os.path.join(ROOT, "..", "..", "vesin")

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
            "-DVESIN_TORCH=ON",
        ]

        CUDA_HOME = os.environ.get("CUDA_HOME")

        if CUDA_HOME is not None:
            cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}")
            cmake_options.append("-DVESIN_ENABLE_CUDA=ON")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--target", "install"],
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
        shutil.copytree(os.path.join(ROOT, "..", "..", "vesin"), "lib")

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

    install_requires = []
    forced_torch_version = os.environ.get("VESIN_TORCH_BUILD_WITH_TORCH_VERSION")
    if forced_torch_version is not None:
        install_requires.append(f"torch =={forced_torch_version}")
    else:
        install_requires.append("torch >=2.3")

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
