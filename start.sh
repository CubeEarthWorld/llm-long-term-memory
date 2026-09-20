#!/usr/bin/env bash
# === Launcher (macOS / Linux) — creates .venv, installs dependencies, starts the web app ===
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -x ".venv/bin/python" ] || "$PY" -m venv .venv
source .venv/bin/activate
# Install only when requirements.txt differs from the copy stored beside the venv:
# a fully-installed app then launches offline (set -e would abort on a failed pip),
# and editing requirements.txt correctly re-triggers the install.
if ! cmp -s requirements.txt .venv/.deps_req; then
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  cp requirements.txt .venv/.deps_req
fi
python server.py
