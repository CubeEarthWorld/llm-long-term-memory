@echo off
REM === Build target\release\ltm.exe (used by start.bat and run_experiment.bat) ===
REM Uses the GPU (CUDA) for the embedding model when the CUDA toolkit is installed,
REM and at most 80%% of the CPU cores while compiling.
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

where cargo >nul 2>&1
if errorlevel 1 (
  echo [error] Rust is not installed. Install it from https://rustup.rs and re-run.
  exit /b 1
)

REM llama.cpp is compiled with CMake + libclang; take both from Visual Studio Build Tools.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VS=%%I"
if defined VS (
  set "PATH=%PATH%;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
  if not defined LIBCLANG_PATH set "LIBCLANG_PATH=%VS%\VC\Tools\Llvm\x64\bin"
)

set /a JOBS=%NUMBER_OF_PROCESSORS%*8/10
if %JOBS% LSS 1 set JOBS=1
set "CMAKE_BUILD_PARALLEL_LEVEL=%JOBS%"

set "FEATURES="
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" if defined VS (
  REM CUDA needs the MSVC environment and the Ninja generator (Build Tools ship no CUDA toolset).
  call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
  set "PATH=!PATH!;%CUDA_PATH%\bin"
  set "CMAKE_GENERATOR=Ninja"
  set "FEATURES=--features cuda"
  echo [build] CUDA found: the embedding model will run on the GPU
)

echo [build] compiling (first build takes several minutes) ...
cargo build --release -j %JOBS% %FEATURES%
if errorlevel 1 (
  echo [error] build failed. See the messages above.
  exit /b 1
)
exit /b 0
