@echo off
REM === Double-click launcher (Windows) ===
setlocal enableextensions
cd /d "%~dp0"

REM keep the EmbeddingGemma model inside this folder (portable)
set "HF_HOME=%~dp0model"
set "HF_HUB_CACHE=%~dp0model\hub"
set "SENTENCE_TRANSFORMERS_HOME=%~dp0model"
set "PYTHONUTF8=1"
set "PORT=8501"

REM === automatically stop any previous instance still listening on the port ===
REM This app owns the port; a leftover listener there is its own stale process and
REM would make the server fail to bind, which looks like "start.bat does not launch".
echo [cleanup] checking for a previous instance on port %PORT% ...
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  echo   stopping old process PID %%P
  taskkill /F /PID %%P >nul 2>&1
)

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
  echo [setup] installing dependencies; first run downloads torch etc, please wait ...
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
echo [run] starting the web app at http://localhost:%PORT%  -  close this window to stop
echo [note] first launch loads the embedding model and can take 10-30s;
echo        the page shows "loading" until it is ready - this is normal.
python server.py

echo.
echo [stopped] the server has exited. See any message above.
pause
