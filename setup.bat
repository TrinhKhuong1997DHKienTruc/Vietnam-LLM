@echo off
REM Fin-o1-8B Model Demo Setup Script
REM For Windows users

echo 🚀 Fin-o1-8B Model Demo Setup
echo ================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH.
    echo 💡 Please install Python 3.8+ from https://python.org
    echo 💡 Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% detected

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv fin_o1_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call fin_o1_env\Scripts\activate.bat

REM Upgrade pip
echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 🔧 Installing dependencies...
pip install -r requirements.txt

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 To run the demo:
echo    fin_o1_env\Scripts\activate.bat
echo    python fin_o1_simple.py
echo.
echo 💡 For interactive chat:
echo    python fin_o1_demo.py
echo.
echo 🎯 Happy exploring with Fin-o1-8B!
pause