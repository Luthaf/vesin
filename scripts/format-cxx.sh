#!/usr/bin/env bash

set -eu

EXTRA_CLANG_FORMAT_ARGS="-i"

for arg in "$@"; do
  if [[ "$arg" == "--check" ]]; then
    EXTRA_CLANG_FORMAT_ARGS="--dry-run --Werror"
    break
  fi
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

echo "using $(clang-format --version) from $(which clang-format)"
git ls-files '*.cpp' '*.c' '*.hpp' '*.h' '*.cu' '*.cuh' | xargs -L 1 clang-format $EXTRA_CLANG_FORMAT_ARGS
