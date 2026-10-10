@echo off
REM === Double-click: replay the seed scenario headlessly and save the results ===
setlocal
cd /d "%~dp0"
call build.bat
if errorlevel 1 (
  pause
  exit /b 1
)
REM The CUDA build loads cudart / cublas at run time.
if defined CUDA_PATH set "PATH=%PATH%;%CUDA_PATH%\bin"
echo [run] running the seed experiment ...
target\release\ltm.exe run
echo.
echo [done] results saved to data\results.json
pause
