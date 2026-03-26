@echo off
:: STABLE PRO-LAUNCHER (BAREBONES)

echo 🎙️ GHOSTTALKER LAUNCHER (v2.0)
echo.

:: 1. Is Python here?
where python >nul 2>nul
if errorlevel 1 (
    echo [ERR] Python NOT found. Install it first!
    pause
    exit /b
)

:: 2. Setup VENV
if not exist venv (
    echo [+] First-time Setup: Building Neural Engine...
    python -m venv venv
)

:: 3. Run
echo [+] Summoning the Ghost...
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py

:: If it crashes, don't close the window!
if errorlevel 1 (
    echo.
    echo [ERR] The engine stopped. Check the error above.
    pause
)

pause
