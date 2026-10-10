@echo off
REM === Double-click launcher (Windows): build, then start the web app ===
setlocal enableextensions
cd /d "%~dp0"
set "PORT=8501"

REM A leftover instance on the port would make the server fail to bind: stop it
REM (only if it is this app's ltm.exe; 8501 is also Streamlit's default).
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  tasklist /FI "PID eq %%P" | findstr /i "ltm.exe" >nul
  if not errorlevel 1 (
    echo [cleanup] stopping old instance PID %%P
    taskkill /F /PID %%P >nul 2>&1
  )
)

call build.bat
if errorlevel 1 (
  pause
  exit /b 1
)
REM The CUDA build loads cudart / cublas at run time.
if defined CUDA_PATH set "PATH=%PATH%;%CUDA_PATH%\bin"

echo.
echo [run] http://localhost:%PORT%  -  close this window to stop
echo [note] the page shows "loading" until the embedding model is ready
target\release\ltm.exe serve --port %PORT%

echo.
echo [stopped] the server has exited. See any message above.
pause
