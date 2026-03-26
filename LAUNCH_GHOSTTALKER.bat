@echo off
TITLE 🎙️ GhostTalker: The Eternal Echo
:: Matrix Green
COLOR 0A

echo --------------------------------------------------
echo      🎙️ GHOSTTALKER: THE ETERNAL ECHO
echo           Maintainer: shantoshdurai
echo --------------------------------------------------
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERR] Python is NOT installed or NOT in your PATH.
    echo [!] Download from: https://www.python.org/downloads/
    pause
    exit /b
)

:: 2. Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [!] ALERT: Virtual environment 'venv' missing.
    echo [+] Starting First-time Setup: Building Neural Engine (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo [ERR] Failed to create virtual environment. 
        pause
        exit /b
    )
    echo.
    echo [+] Environment Created. Installing Pro Dependencies...
    echo [+] This will take a few minutes (downloading Torch/CUDA/XTTS)...
    echo.
    call .\venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERR] Dependency installation failed! 
        pause
        exit /b
    )
    echo.
    echo [+] SETUP COMPLETE. Engine Stabilized.
    echo.
)

:: 3. Launch Browser
echo [+] Initiating Neural Interface...
start http://127.0.0.1:9988

:: 4. Ignite GhostTalker Engine
echo [+] Starting Flask Backend on CUDA...
echo.
:: Explicitly use the python inside the venv
.\venv\Scripts\python.exe app.py

if errorlevel 1 (
    echo [ERR] Application crashed or stopped unexpectedly.
    pause
)

pause
