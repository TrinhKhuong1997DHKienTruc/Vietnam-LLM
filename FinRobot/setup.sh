#!/bin/bash

echo "========================================"
echo "    FinRobot Setup for Linux/macOS"
echo "========================================"
echo

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python3 is not installed or not in PATH"
    echo "Please install Python 3.13+ from https://python.org"
    exit 1
fi

# Check if python3-venv is available
echo "Checking for python3-venv..."
python3 -c "import venv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing python3-venv..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-venv
    elif command -v brew &> /dev/null; then
        brew install python3
    else
        echo "Please install python3-venv manually"
        exit 1
    fi
fi

echo
echo "Creating virtual environment..."
python3 -m venv finrobot_env
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo
echo "Activating virtual environment..."
source finrobot_env/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo
echo "Installing dependencies..."
pip install -r requirements_simple.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo
echo "========================================"
echo "    Setup completed successfully!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Configure your API keys in config_api_keys"
echo "2. Configure OpenAI in OAI_CONFIG_LIST"
echo "3. Run: python simple_demo.py"
echo "4. Generate reports: python generate_reports.py"
echo
echo "To activate the environment later, run:"
echo "source finrobot_env/bin/activate"
echo