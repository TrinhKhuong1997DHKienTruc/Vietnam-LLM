#!/bin/bash

# Fin-o1-8B Model Demo Setup Script
# For Linux and macOS users

set -e

echo "🚀 Fin-o1-8B Model Demo Setup"
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    echo "💡 On Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "💡 On macOS: brew install python3"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv fin_o1_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source fin_o1_env/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "🔧 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To run the demo:"
echo "   source fin_o1_env/bin/activate"
echo "   python fin_o1_simple.py"
echo ""
echo "💡 For interactive chat:"
echo "   python fin_o1_demo.py"
echo ""
echo "🎯 Happy exploring with Fin-o1-8B!"