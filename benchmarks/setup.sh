#!/usr/bin/env bash
#
# Set up the environment to run the benchmarks: install the Python dependencies,
# then build LAMMPS from the official upstream sources with OpenMP enabled and
# install its Python module.
#
# This must run inside an activated virtual environment, or with PYTHON pointing
# to the python executable of one. Other environment variables:
#
# - LAMMPS_VERSION      git tag/branch of LAMMPS to build
# - LAMMPS_BUILD_DIR    where to clone and build LAMMPS (default: benchmarks/build)
# - OpenMP_ROOT         where to find OpenMP (defaults to Homebrew's libomp on macOS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
LAMMPS_VERSION="${LAMMPS_VERSION:-stable_30Sep2026_update1}"
LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-$SCRIPT_DIR/build}"

if ! "$PYTHON" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)'; then
    echo "error: $PYTHON is not running inside a virtual environment" >&2
    exit 1
fi
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"

"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

LAMMPS_SOURCE="$LAMMPS_BUILD_DIR/lammps-$LAMMPS_VERSION"
if [ ! -d "$LAMMPS_SOURCE" ]; then
    git clone --depth 1 --branch "$LAMMPS_VERSION" https://github.com/lammps/lammps "$LAMMPS_SOURCE"
fi

CMAKE_BUILD="$LAMMPS_SOURCE/build"
EXTRA_CMAKE_ARGS=()

# Apple clang does not ship OpenMP, and torch (which the benchmarks import) ships
# its own libomp on macOS. Loading two copies of libomp in the same process aborts
# with "OMP: Error #15", so build LAMMPS against the copy from torch.
if [ "$(uname)" = "Darwin" ]; then
    TORCH_DIR="$("$PYTHON" -c 'import os, torch; print(os.path.dirname(torch.__file__))')"

    # torch's libomp has the install name of the machine it was built on
    # (/opt/llvm-openmp/lib/libomp.dylib). Link against a copy using an @rpath
    # install name instead, and point the rpath to torch's directory.
    OPENMP_DIR="$CMAKE_BUILD/torch-openmp"
    mkdir -p "$OPENMP_DIR/include" "$OPENMP_DIR/lib"
    cp "$TORCH_DIR/include/omp.h" "$OPENMP_DIR/include/"
    cp "$TORCH_DIR/lib/libomp.dylib" "$OPENMP_DIR/lib/"
    install_name_tool -id @rpath/libomp.dylib "$OPENMP_DIR/lib/libomp.dylib"

    EXTRA_CMAKE_ARGS+=(
        -DOpenMP_ROOT="$OPENMP_DIR"
        # the build directory rpath would point to the copy of libomp above
        -DCMAKE_SKIP_BUILD_RPATH=ON
        # the LAMMPS python package is installed next to torch in site-packages
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,@loader_path/../torch/lib"
    )
fi

# Only the core of LAMMPS and the OPENMP package are needed by the benchmarks.
# MPI is disabled since the benchmarks run on a single process.
cmake --fresh -S "$LAMMPS_SOURCE/cmake" -B "$CMAKE_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_MPI=OFF \
    -DBUILD_OMP=ON \
    -DPKG_OPENMP=ON \
    -DPython_EXECUTABLE="$PYTHON" \
    ${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"}
cmake --build "$CMAKE_BUILD" --parallel

# Build the wheel with LAMMPS' own script, but install it ourselves: the script
# falls back to installing in the user or system site-packages on failure.
if [ "$(uname)" = "Darwin" ]; then
    LAMMPS_LIBRARY="$CMAKE_BUILD/liblammps.dylib"
else
    LAMMPS_LIBRARY="$CMAKE_BUILD/liblammps.so"
fi

rm -f "$CMAKE_BUILD"/lammps-*.whl
(
    cd "$CMAKE_BUILD"
    "$PYTHON" "$LAMMPS_SOURCE/python/install.py" --noinstall \
        --package "$LAMMPS_SOURCE/python/lammps" \
        --lib "$LAMMPS_LIBRARY" \
        --versionfile "$LAMMPS_SOURCE/src/version.h" \
        --wheeldir "$CMAKE_BUILD"
)
"$PYTHON" -m pip install --force-reinstall --no-deps "$CMAKE_BUILD"/lammps-*.whl

"$PYTHON" - << 'EOF'
import lammps
import torch

# make sure torch and LAMMPS can use OpenMP in the same process
torch.ones(100, 100) @ torch.ones(100, 100)
lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-pk", "omp", "2"])
if lmp.extract_setting("nthreads") != 2:
    raise SystemExit("error: LAMMPS was built without OpenMP support")
print(f"LAMMPS {lmp.version()} installed, with OpenMP support")
EOF
