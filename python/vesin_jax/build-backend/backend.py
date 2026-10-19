# This is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies on jaxlib/cmake when building the wheel and not the sdist
import os

from setuptools import build_meta


ROOT = os.path.realpath(os.path.dirname(__file__))

FORCED_JAXLIB_VERSION = os.environ.get("VESIN_JAX_BUILD_WITH_JAXLIB_VERSION")
if FORCED_JAXLIB_VERSION is not None:
    JAXLIB_DEP = f"jaxlib =={FORCED_JAXLIB_VERSION}"
else:
    JAXLIB_DEP = "jaxlib >=0.10"

# ==================================================================================== #
#                   Build backend functions definition                                 #
# ==================================================================================== #

# Use the default version of these
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


# Special dependencies to build the wheels
def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + ["cmake", JAXLIB_DEP]
