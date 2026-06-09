@echo off
REM === Double-click: run the 10-utterance experiment headlessly and save results ===
setlocal
cd /d "%~dp0"
set "HF_HOME=%~dp0model"
set "HF_HUB_CACHE=%~dp0model\hub"
set "SENTENCE_TRANSFORMERS_HOME=%~dp0model"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_ok" (
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo [error] dependency install failed.
    pause
    exit /b 1
  )
  echo ok> ".venv\.deps_ok"
)

echo [run] running the LLM Long-Term Memory seed experiment ...
python cli.py
echo.
echo [done] results saved to data\results.json
pause
