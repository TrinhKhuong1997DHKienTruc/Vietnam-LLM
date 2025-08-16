#!/bin/bash

# Fin-o1-8B Setup Script
# This script sets up the environment for running the Fin-o1-8B model

echo "🚀 Setting up Fin-o1-8B Model Environment"
echo "=========================================="

# Check Python version
echo "🔍 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
echo "🔍 Checking pip availability..."
python3 -m pip --version
if [ $? -ne 0 ]; then
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check transformers
echo "🔍 Checking transformers installation..."
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x run_fin_o1.py
chmod +x financial_demo.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Quick start options:"
echo "  1. Interactive chat: python3 run_fin_o1.py --interactive"
echo "  2. Financial demo:   python3 financial_demo.py"
echo "  3. Quick test:       python3 financial_demo.py --quick-test"
echo ""
echo "💡 For memory optimization, use:"
echo "  - 8-bit:  python3 run_fin_o1.py --eight-bit --interactive"
echo "  - 4-bit:  python3 run_fin_o1.py --four-bit --interactive"
echo ""
echo "📚 See README.md for detailed usage instructions."