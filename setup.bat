@echo off
REM FinBERT AAPL Price Prediction Demo - Windows Setup Script
echo ================================================================
echo FinBERT AAPL Price Prediction Demo - Windows Setup
echo ================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv finbert_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call finbert_env\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM Test FinBERT
echo.
echo Testing FinBERT model...
python test_finbert.py
if errorlevel 1 (
    echo WARNING: FinBERT test failed, but continuing...
)

echo.
echo ================================================================
echo Setup completed successfully!
echo ================================================================
echo.
echo To run the demo:
echo 1. Activate the virtual environment: finbert_env\Scripts\activate
echo 2. Run the demo: python aapl_price_prediction_demo.py
echo.
echo To deactivate the virtual environment later:
echo deactivate
echo.
pause
