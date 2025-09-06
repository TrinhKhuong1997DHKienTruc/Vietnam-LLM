@echo off
echo ========================================
echo    FinRobot Setup for Windows
echo ========================================
echo.

echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.13+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv finrobot_env
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call finrobot_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements_simple.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Configure your API keys in config_api_keys
echo 2. Configure OpenAI in OAI_CONFIG_LIST
echo 3. Run: python simple_demo.py
echo 4. Generate reports: python generate_reports.py
echo.
echo To activate the environment later, run:
echo finrobot_env\Scripts\activate.bat
echo.
pause