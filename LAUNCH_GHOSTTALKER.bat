@echo off
setlocal enabledelayedexpansion
title GhostTalker Launcher
cd /d %~dp0

echo.
echo  ======================================================
echo            GHOSTTALKER VOICE CLONING TOOL
echo  ======================================================
echo.

:: ── Python check ──────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.10, 3.11, or 3.12 from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause & exit /b 1
)

:: Warn if Python 3.13+ (not yet supported by all ML packages)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=2 delims=." %%m in ("!PYVER!") do set PYMINOR=%%m
if !PYMINOR! GEQ 13 (
    echo  [WARN] Python !PYVER! detected. Python 3.13+ may have issues with some packages.
    echo  Recommended: Python 3.10, 3.11, or 3.12
    echo.
)

echo  [*] Python !PYVER! detected.

:: ── Virtual environment ────────────────────────────────────
if not exist venv (
    echo  [*] Creating virtual environment (first time only)...
    python -m venv venv
)

echo  [*] Activating virtual environment...
call venv\Scripts\activate

:: ── Pip ────────────────────────────────────────────────────
echo  [*] Checking pip...
python -m pip install --upgrade pip --quiet

:: ── PyTorch (CUDA 12.1 for NVIDIA GPUs) ───────────────────
echo  [*] Installing PyTorch with CUDA 12.1 support...
echo      (skipped if already installed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

:: ── Project dependencies ───────────────────────────────────
echo  [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet

:: ── First-run model notice ─────────────────────────────────
if not exist "%USERPROFILE%\.cache\huggingface\hub\models--SWivid--F5-TTS" (
    echo.
    echo  ┌─────────────────────────────────────────────────────┐
    echo  │  FIRST RUN — Model Download Required (~1.7 GB)      │
    echo  │                                                      │
    echo  │  • F5-TTS voice model    ~1.35 GB                   │
    echo  │  • Whisper ASR model      ~290 MB                   │
    echo  │  • Vocoder                 ~54 MB                   │
    echo  │                                                      │
    echo  │  Downloads happen once and are cached automatically. │
    echo  │  This may take 5-10 minutes on first launch.        │
    echo  └─────────────────────────────────────────────────────┘
    echo.
)

:: ── Launch ─────────────────────────────────────────────────
echo  [*] Starting GhostTalker...
echo  [*] Open your browser at:  http://127.0.0.1:7860
echo  [*] Press Ctrl+C to stop.
echo.

:: Open browser after 5 seconds (gives the server time to start)
start "" cmd /c "timeout /t 5 >nul && start http://127.0.0.1:7860"

python app.py

echo.
echo  [*] GhostTalker has stopped.
pause
