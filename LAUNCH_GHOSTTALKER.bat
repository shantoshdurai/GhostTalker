@echo off
TITLE 🎙️ GhostTalker: The Eternal Echo
COLOR 0D

echo --------------------------------------------------
echo      🎙️ GHOSTTALKER: THE ETERNAL ECHO
echo           Maintainer: shantoshdurai
echo --------------------------------------------------
echo.

:: 1. Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [!] ALERT: Virtual environment 'venv' missing.
    echo [+] Starting First-time Setup: Building Neural Engine (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo [ERR] Python not found in Path. Please install Python!
        pause
        exit /b
    )
    echo.
    echo [+] Environment Created. Installing Pro Dependencies...
    echo [+] This might take a few minutes (downloading Torch/CUDA/XTTS)...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo.
    echo [+] SETUP COMPLETE. Engine Stabilized.
    echo.
)

:: 2. Launch Browser (safeguard)
echo [+] Initiating Neural Interface...
start http://127.0.0.1:9988

:: 3. Ignite GhostTalker Engine
echo [+] Starting Flask Backend on CUDA...
echo.
call venv\Scripts\python.exe app.py

pause
