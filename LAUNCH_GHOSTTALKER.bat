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
    echo [!] Recommendation: Run 'python -m venv venv' and install requirements.txt.
    pause
    exit /b
)

:: 2. Launch Browser (safeguard)
echo [+] Initiating Neural Interface...
start http://127.0.0.1:9988

:: 3. Ignite GhostTalker Engine
echo [+] Starting Flask Backend on CUDA...
echo.
call venv\Scripts\python.exe app.py

pause
