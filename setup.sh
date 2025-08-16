#!/bin/bash

echo "Setting up environment for Fin-o1-14B model..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Create models directory
mkdir -p models

echo "Setup complete!"
echo ""
echo "To run the model:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the simple version: python simple_run.py"
echo "3. Or run the full version: python run_fin_o1_14b.py"
echo ""
echo "Note: The first run will download the 14B parameter model, which may take a while."