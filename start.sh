#!/usr/bin/env bash
# === Launcher (Linux) — same as start.command ===
set -e
cd "$(dirname "$0")"
export HF_HOME="$(pwd)/model"
export HF_HUB_CACHE="$(pwd)/model/hub"
export SENTENCE_TRANSFORMERS_HOME="$(pwd)/model"
export PYTHONUTF8=1
PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -x ".venv/bin/python" ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python server.py
