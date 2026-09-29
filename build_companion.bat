@echo off
REM ================================================================
REM  Build / rebuild the "momo" Ollama model from companion.md.
REM  Run this every time you edit companion.md.
REM
REM  Usage:
REM    build_companion.bat            normal rebuild
REM    build_companion.bat --clean    delete the model first
REM ================================================================

setlocal enabledelayedexpansion

REM ---- Configuration -------------------------------------------
set BASE_MODEL=qwen2.5:3b
set MODEL_NAME=momo
set MD_FILE=companion.md
set MODELFILE=Modelfile

REM ---- Parameters (tuned for HP Pavilion, Ryzen 5 3550H, 16 GB) -
set TEMPERATURE=0.3
set TOP_P=0.9
set TOP_K=40
set REPEAT_PENALTY=1.1
set NUM_CTX=4096
set NUM_PREDICT=180
set NUM_THREAD=6
set NUM_GPU=20
REM --------------------------------------------------------------

if not exist "%MD_FILE%" (
    echo.
    echo Error: %MD_FILE% not found in this directory.
    echo.
    exit /b 1
)

if /I "%~1"=="--clean" (
    echo Removing existing model "%MODEL_NAME%"...
    ollama rm %MODEL_NAME% >nul 2>&1
    echo.
)

echo Generating Modelfile from %MD_FILE%...

(
    echo FROM %BASE_MODEL%
    echo(
    echo SYSTEM """
    type "%MD_FILE%"
    echo """
    echo(
    echo PARAMETER temperature %TEMPERATURE%
    echo PARAMETER top_p %TOP_P%
    echo PARAMETER top_k %TOP_K%
    echo PARAMETER repeat_penalty %REPEAT_PENALTY%
    echo PARAMETER num_ctx %NUM_CTX%
    echo PARAMETER num_predict %NUM_PREDICT%
    echo PARAMETER num_thread %NUM_THREAD%
    echo PARAMETER num_gpu %NUM_GPU%
) > "%MODELFILE%"

echo Modelfile written.
echo.
echo Building Ollama model "%MODEL_NAME%" from %BASE_MODEL%...
echo.

ollama create %MODEL_NAME% -f %MODELFILE%

if errorlevel 1 (
    echo.
    echo Build failed.
    echo.
    exit /b 1
)

echo.
echo Done.
echo Run it with:
echo     ollama run %MODEL_NAME%
echo.

endlocal