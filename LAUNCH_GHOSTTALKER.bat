@echo off
setlocal
title GhostTalker Launcher

:: Move to script directory
cd /d %~dp0

echo ======================================================
echo           GHOSTTALKER VOICE CLONING TOOL
echo ======================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Setup Virtual Environment
if not exist venv (
    echo [*] Creating virtual environment...
    python -m venv venv
)

echo [*] Activating venv...
call venv\Scripts\activate

:: Update Pip
echo [*] Updating pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA 12.1
echo [*] Ensuring Torch (CUDA 12.1) is installed...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install other dependencies
echo [*] Installing GhostTalker dependencies...
pip install -r requirements.txt

:: Launch the App
echo [*] Starting the Web UI...
python app.py

pause
