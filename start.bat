@echo off
REM === Double-click launcher (Windows) ===
setlocal
cd /d "%~dp0"

REM keep the EmbeddingGemma model inside this folder (portable)
set "HF_HOME=%~dp0model"
set "HF_HUB_CACHE=%~dp0model\hub"
set "SENTENCE_TRANSFORMERS_HOME=%~dp0model"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [error] Python not found. Install Python 3.10+ from https://python.org and re-run.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_ok" (
  echo [setup] installing dependencies (first run downloads torch etc; please wait) ...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo [error] dependency install failed. Check your internet connection and re-run.
    pause
    exit /b 1
  )
  echo ok> ".venv\.deps_ok"
)

echo.
echo [run] starting the web app at http://localhost:8501  (close this window to stop)
python server.py
pause
