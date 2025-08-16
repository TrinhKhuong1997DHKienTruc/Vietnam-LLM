#!/bin/bash

echo "=== Fin-o1-14B Setup Script ==="
echo "This script will set up the environment to run the Fin-o1-14B model"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "pip3 found ✓"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected ✓"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: No NVIDIA GPU detected. The model will run on CPU (very slow for 14B parameters)"
    echo "Consider using Google Colab or a cloud GPU instance for better performance."
    echo ""
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv fin_o1_env

# Activate virtual environment
echo "Activating virtual environment..."
source fin_o1_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run the model:"
echo "1. Activate the virtual environment: source fin_o1_env/bin/activate"
echo "2. Run the model: python run_fin_o1.py"
echo ""
echo "Example commands:"
echo "  # Interactive mode"
echo "  python run_fin_o1.py"
echo ""
echo "  # Single prompt mode"
echo "  python run_fin_o1.py --prompt 'What is the difference between stocks and bonds?'"
echo ""
echo "  # Use 4-bit quantization (saves more memory)"
echo "  python run_fin_o1.py --4bit"
echo ""
echo "Note: The first run will download the model (~28GB), which may take some time."
echo "Make sure you have sufficient disk space and a stable internet connection."