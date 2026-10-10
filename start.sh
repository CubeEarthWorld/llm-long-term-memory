#!/usr/bin/env bash
# === Launcher (macOS / Linux): build, then start the web app ===
set -e
cd "$(dirname "$0")"
./build.sh
exec ./target/release/ltm serve --port "${PORT:-8501}"
