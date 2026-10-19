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
    # here, and the wheel will be compatible with any Python >=3.10. This is
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
        import jaxlib

        source_dir = os.path.join(ROOT, "lib")
        if not os.path.exists(source_dir):
            # we are building from a checkout
            source_dir = os.path.join(ROOT, "..", "..", "vesin-jax")

        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(build_dir, "install")

        os.makedirs(build_dir, exist_ok=True)

        xla_include_dir = os.path.join(os.path.dirname(jaxlib.__file__), "include")
        assert os.path.exists(xla_include_dir)

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={VESIN_BUILD_TYPE}",
            f"-DXLA_INCLUDE_DIR={xla_include_dir}",
        ]

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

        # copy the shared library inside the package, without any additional
        # files or directory structure
        for candidate in glob.glob(os.path.join(install_dir, "lib", "libvesin_jax.*")):
            shutil.copy(candidate, os.path.join(self.build_lib, "vesin_jax"))

        for candidate in glob.glob(os.path.join(install_dir, "bin", "vesin_jax.*")):
            shutil.copy(candidate, os.path.join(self.build_lib, "vesin_jax"))

        if not glob.glob(os.path.join(self.build_lib, "vesin_jax", "*vesin_jax*")):
            raise Exception("could not find the vesin_jax shared library")


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
        shutil.copytree(
            os.path.join(ROOT, "..", "..", "vesin-jax"), os.path.join("lib")
        )

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
            f"https://github.com/metatensor/gpu-lite/archive/{commit}.tar.gz",
            gpulite_archive,
        )

        # run original sdist
        super().run()

        # cleanup
        shutil.rmtree("lib")


if __name__ == "__main__":
    install_requires = [
        "jax >=0.10",
        "jaxlib >=0.10",
    ]

    setup(
        version=open("VERSION").read().strip(),
        install_requires=install_requires,
        ext_modules=[
            # only declare the extension, it is built & copied as required by cmake
            # in the build_ext command
            Extension(name="vesin_jax", sources=[]),
        ],
        cmdclass={
            "sdist": sdist_with_lib,
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
        },
    )
