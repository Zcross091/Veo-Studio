@echo off
title Veo Studio
cd /d "%~dp0"

echo ==========================================================
echo               Launching Veo Studio
echo ==========================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please download and install Python 3.10+ from https://python.org
    echo and remember to check "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

:: 2. Set up virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate and verify dependencies
call .venv\Scripts\activate.bat
echo [*] Checking dependencies...
pip install -r requirements.txt --quiet

:: 4. Start the all-in-one launcher
echo [*] Starting web interface...
python run.py

pause

