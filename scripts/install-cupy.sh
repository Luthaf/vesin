#!/usr/bin/env bash
# Install a matching cupy wheel for the local CUDA reported by nvidia-smi.
# Exits if nvidia-smi is not present or no matching cupy package could be installed.
set -euo pipefail

# require nvidia-smi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    exit 0
fi

# try to read CUDA version from nvidia-smi output
cuda_ver=$(nvidia-smi 2>/dev/null | grep -i 'cuda version' | sed -E 's/.*CUDA Version: *([0-9]+).*/\1/i' || true)

# fallback: try nvcc if present
if [ -z "$cuda_ver" ] && command -v nvcc >/dev/null 2>&1; then
    cuda_ver=$(nvcc --version 2>/dev/null | sed -nE 's/.*release ([0-9]+).*/\1/p' || true)
fi

if [ -z "$cuda_ver" ]; then
    err "Could not detect CUDA version from nvidia-smi or nvcc. Exiting."
    exit 1
fi

python -m pip install "cupy-cuda${cuda_ver}x"
