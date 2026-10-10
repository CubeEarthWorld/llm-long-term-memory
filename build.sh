#!/usr/bin/env bash
# === Build target/release/ltm (used by start.sh) ===
# Uses the GPU for the embedding model when available (CUDA) and at
# most 80% of the CPU cores while compiling.
set -e
cd "$(dirname "$0")"
command -v cargo >/dev/null 2>&1 || { echo "[error] Rust is not installed: https://rustup.rs"; exit 1; }
CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
JOBS=$(( CORES * 8 / 10 )); [ "$JOBS" -ge 1 ] || JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
FEATURES=""
if command -v nvcc >/dev/null 2>&1; then
  FEATURES="--features cuda"
  echo "[build] CUDA found: the embedding model will run on the GPU"
fi
echo "[build] compiling (first build takes several minutes) ..."
cargo build --release -j "$JOBS" $FEATURES
