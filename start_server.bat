@echo off
title Netram AI Deepfake Shield - Native Inference Engine
echo ======================================================================
echo          Netram AI Deepfake Shield - Native Inference Engine
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Starting Netram Engine with virtual environment...
    .venv\Scripts\python.exe extension_server.py
) else (
    echo [INFO] Starting Netram Engine with system Python...
    python extension_server.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server terminated with error code %ERRORLEVEL%.
    pause
)
