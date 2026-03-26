@echo off
TITLE GhostTalker Debug Launcher
COLOR 0A

echo --------------------------------------------------
echo      🎙️ GHOSTTALKER DEBUG LAUNCHER
echo --------------------------------------------------
echo.

echo [+] Checking for Python...
python --version
if errorlevel 1 (
    echo [ERR] Python.exe not found.
    pause
    exit /b
)

echo [+] Checking for Virtual Environment...
if not exist "venv\Scripts\python.exe" (
    echo [!] ALERT: Virtual environment missing. Creating now...
    python -m venv venv
    if errorlevel 1 (
        echo [ERR] Failed to create venv.
        pause
        exit /b
    )
)

echo [+] Activating Engine...
call venv\Scripts\activate.bat

echo [+] Starting App...
python app.py

if errorlevel 1 (
    echo [ERR] App failed to start.
    pause
)

pause
