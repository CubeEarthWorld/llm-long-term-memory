@echo off
REM === Double-click: run the seed experiment headlessly and save results ===
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"

REM fc compares requirements.txt with the copy saved beside the venv: a missing or
REM stale copy re-runs the install, so editing requirements.txt is picked up.
fc /b requirements.txt ".venv\.deps_req" >nul 2>&1
if errorlevel 1 (
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo [error] dependency install failed.
    pause
    exit /b 1
  )
  copy /y requirements.txt ".venv\.deps_req" >nul
)

echo [run] running the LLM Long-Term Memory seed experiment ...
python cli.py
echo.
echo [done] results saved to data\results.json
pause
