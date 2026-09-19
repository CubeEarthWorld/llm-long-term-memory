#!/usr/bin/env bash
# === Launcher (macOS / Linux) — creates .venv, installs dependencies, starts the web app ===
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -x ".venv/bin/python" ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python server.py
