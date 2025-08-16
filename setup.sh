#!/bin/bash

echo "Setting up Fin-o1-14B Model Environment"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $python_version is installed, but Python $required_version+ is required."
    exit 1
fi

echo "Python version: $python_version ✓"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "pip3 found ✓"

# Check available memory
if command -v free &> /dev/null; then
    total_mem=$(free -g | awk 'NR==2{print $2}')
    echo "Total system memory: ${total_mem}GB"
    
    if [ "$total_mem" -lt 16 ]; then
        echo "Warning: Less than 16GB RAM detected. The model may not run properly."
        echo "Consider using a machine with more memory or using quantization."
    fi
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected ✓"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read -r name memory; do
        echo "GPU: $name, Memory: ${memory}MB"
    done
else
    echo "No NVIDIA GPU detected - will run on CPU (will be very slow for 14B model)"
fi

# Create virtual environment (optional)
read -p "Would you like to create a virtual environment? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv fin_o1_env
    echo "Virtual environment created. Activate it with:"
    echo "source fin_o1_env/bin/activate"
    echo ""
    echo "Then install dependencies with:"
    echo "pip install -r requirements.txt"
else
    echo "Installing dependencies globally..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To run the model:"
echo "1. If you created a virtual environment, activate it:"
echo "   source fin_o1_env/bin/activate"
echo ""
echo "2. Run the simple version (recommended for beginners):"
echo "   python3 simple_run.py"
echo ""
echo "3. Or run the advanced version with quantization:"
echo "   python3 run_model.py"
echo ""
echo "Note: The first run will download the model (~30GB) which may take some time."
echo "Make sure you have a stable internet connection and sufficient disk space."