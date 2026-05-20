@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   GEMM Matmul -- Nsight Compute Profiles
echo ========================================
echo.

call conda activate ainfra
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env 'ainfra'
    pause
    exit /b 1
)

REM Create timestamped subfolder
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,4%
set PROF_DIR=profiles\%TIMESTAMP%
if not exist "%PROF_DIR%\" mkdir "%PROF_DIR%"

echo Output: %PROF_DIR%\
echo.

set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\ncu.bat"
set PYTHON=python
set SCRIPT=profile_kernel.py

REM -s 5: skip 5 warmup calls, profile first timing call
REM -c 1: capture only 1 kernel invocation

echo [1/6] Profiling Naive Matmul...
call %NCU% --set full -k regex:matmul_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_01_naive %PYTHON% %SCRIPT% matmul_kernel
echo.

echo [2/6] Profiling Shared Memory Matmul...
call %NCU% --set full -k regex:matmul_shared_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_02_shared %PYTHON% %SCRIPT% matmul_shared_kernel
echo.

echo [3/6] Profiling Float4 + Shared Memory Matmul...
call %NCU% --set full -k regex:matmul_shared_float4_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_03_float4 %PYTHON% %SCRIPT% matmul_shared_float4_kernel
echo.

echo [4/6] Profiling Register Tiling Matmul...
call %NCU% --set full -k regex:matmul_register_tiling_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_04_regtile %PYTHON% %SCRIPT% matmul_register_tiling_kernel
echo.

echo [5/6] Profiling DB Async (cp.async Double Buffering) ...
call %NCU% --set full -k regex:gemm_db_async_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_05_dbasync %PYTHON% %SCRIPT% gemm_db_async_kernel
echo.

echo [6/6] Profiling WMMA TF32 Tensor Core ...
call %NCU% --set full -k regex:gemm_wmma_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_06_wmma %PYTHON% %SCRIPT% gemm_wmma_kernel
echo.

echo ========================================
echo   All profiles complete.
echo   Open with: ncu-ui %PROF_DIR%\*.ncu-rep
echo ========================================
pause
