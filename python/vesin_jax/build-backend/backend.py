# This is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies on jax/cmake when building the wheel and not the sdist
import os

from setuptools import build_meta


# The native library only uses the XLA FFI, which stays compatible with older
# versions of its API. A wheel built against a given version of jax works with
# all newer versions, so the release wheels are built against the oldest jax we
# support, by setting this environment variable.
FORCED_JAX_VERSION = os.environ.get("VESIN_JAX_BUILD_WITH_JAX_VERSION")
if FORCED_JAX_VERSION is not None:
    JAX_DEP = f"jax =={FORCED_JAX_VERSION}"
else:
    JAX_DEP = "jax >=0.6.2"

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
    return defaults + ["cmake", JAX_DEP]
