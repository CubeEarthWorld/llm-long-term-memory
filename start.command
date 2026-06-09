#!/usr/bin/env bash
# === Double-click launcher (macOS / Linux) ===
# On macOS this file is double-clickable. On Linux: chmod +x start.command && ./start.command
set -e
cd "$(dirname "$0")"

export HF_HOME="$(pwd)/model"
export HF_HUB_CACHE="$(pwd)/model/hub"
export SENTENCE_TRANSFORMERS_HOME="$(pwd)/model"
export PYTHONUTF8=1

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] creating virtual environment .venv ..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate

echo "[setup] installing dependencies (first run downloads torch etc; please wait) ..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "[run] starting the web app — open http://localhost:8501"
python server.py
