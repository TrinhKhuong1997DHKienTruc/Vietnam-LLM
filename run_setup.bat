@echo off
echo ============================================================
echo NVDA Stock Prediction Setup for Windows
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found! Starting setup...
echo.

REM Run the setup script
python setup.py

echo.
echo Setup process completed!
echo Check the demo_reports folder for generated files.
echo.
pause
