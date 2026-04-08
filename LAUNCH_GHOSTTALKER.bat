@echo off
cd /d "%~dp0"
title GhostTalker

echo ======================================================
echo           GHOSTTALKER VOICE CLONING TOOL
echo ======================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.10 or 3.11 from https://python.org
    echo Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

if not exist venv (
    echo [*] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [*] Updating pip...
python -m pip install --upgrade pip --quiet

echo [*] Installing PyTorch CUDA 12.1...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

echo [*] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [*] Starting GhostTalker...
echo [*] Open http://127.0.0.1:7860 in your browser
echo [*] Press Ctrl+C to stop.
echo.

start "" cmd /c "timeout /t 6 >nul && start http://127.0.0.1:7860"
python app.py

pause
