@echo off
setlocal enabledelayedexpansion
title GhostTalker Launcher
cd /d %~dp0

echo.
echo  ======================================================
echo            GHOSTTALKER VOICE CLONING TOOL
echo  ======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.10, 3.11, or 3.12 from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [*] Python %PYVER% detected.

:: Warn if Python 3.13+
for /f "tokens=2 delims=." %%m in ("%PYVER%") do set PYMINOR=%%m
if %PYMINOR% GEQ 13 (
    echo  [WARN] Python 3.13+ may have issues. Recommended: 3.10, 3.11, or 3.12
    echo.
)

:: Create virtual environment if needed
if not exist venv (
    echo  [*] Creating virtual environment (first time only)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate venv
echo  [*] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Upgrade pip
echo  [*] Updating pip...
python -m pip install --upgrade pip --quiet

:: Install PyTorch CUDA 12.1
echo  [*] Installing PyTorch (CUDA 12.1)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

:: Install all other dependencies
echo  [*] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)

:: First run notice
if not exist "%USERPROFILE%\.cache\huggingface\hub\models--SWivid--F5-TTS" (
    echo.
    echo  NOTE - First run will download ~1.7 GB of AI models.
    echo  This happens once and takes 5-10 minutes depending on your internet.
    echo  F5-TTS model: 1.35 GB
    echo  Whisper ASR:   290 MB
    echo  Vocoder:        54 MB
    echo.
)

:: Open browser after delay
echo  [*] Starting GhostTalker...
echo  [*] Your browser will open automatically at http://127.0.0.1:7860
echo  [*] Press Ctrl+C in this window to stop.
echo.
start "" cmd /c "timeout /t 6 >nul && start http://127.0.0.1:7860"

:: Run app
python app.py
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] GhostTalker crashed. See error above.
)

echo.
pause
