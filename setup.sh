#!/bin/bash
# FinBERT AAPL Price Prediction Demo - Linux/macOS Setup Script

echo "================================================================"
echo "FinBERT AAPL Price Prediction Demo - Linux/macOS Setup"
echo "================================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

echo "Python found. Checking version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv finbert_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source finbert_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

# Test FinBERT
echo ""
echo "Testing FinBERT model..."
python test_finbert.py
if [ $? -ne 0 ]; then
    echo "WARNING: FinBERT test failed, but continuing..."
fi

echo ""
echo "================================================================"
echo "Setup completed successfully!"
echo "================================================================"
echo ""
echo "To run the demo:"
echo "1. Activate the virtual environment: source finbert_env/bin/activate"
echo "2. Run the demo: python aapl_price_prediction_demo.py"
echo ""
echo "To deactivate the virtual environment later:"
echo "deactivate"
echo ""
