#!/bin/bash

echo "============================================================"
echo "NVDA Stock Prediction Setup for Linux/macOS"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed"
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Python found! Starting setup..."
echo

# Make the script executable
chmod +x "$0"

# Run the setup script
$PYTHON_CMD setup.py

echo
echo "Setup process completed!"
echo "Check the demo_reports folder for generated files."
echo
