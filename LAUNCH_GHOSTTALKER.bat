@echo off
TITLE 🎙️ GhostTalker Debug Launcher
COLOR 0A

echo --------------------------------------------------
echo      🎙️ GHOSTTALKER DEBUG LAUNCHER
echo --------------------------------------------------
echo.

:: 1. Verify Python
echo [+] Checking for Python...
python --version
if errorlevel 1 (
    echo [ERR] Python.exe not found. Please install Python 3.10+.
    pause
    exit /b
)

:: 2. Setup VENV + Install Requirements
if not exist "venv\Scripts\python.exe" (
    echo [!] ALERT: Virtual environment 'venv' missing. 
    echo [+] Creating virtual environment...
    python -m venv venv
    
    echo [+] Activating environment...
    call venv\Scripts\activate.bat
    
    echo [+] Installing dependencies (this takes a few minutes)...
    echo.
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERR] Dependency installation failed! Check your internet connection.
        pause
        exit /b
    )
    echo.
    echo [+] SETUP COMPLETE. All requirements installed.
) else (
    echo [+] Environment 'venv' detected.
    call venv\Scripts\activate.bat
)

:: 3. Launch the Backend
echo [+] Starting GhostTalker Engine (app.py)...
echo.
python app.py

if errorlevel 1 (
    echo [ERR] GhostTalker crashed or stopped unexpectedly.
    pause
)

pause
