#!/bin/bash

echo "🦙 Fin-o1-8B Setup Script (Virtual Environment)"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if venv module is available
if ! python3 -c "import venv" &> /dev/null; then
    echo "❌ Python venv module not available. Please install python3-venv:"
    echo "   sudo apt install python3-venv"
    exit 1
fi

echo "✅ Python venv module available"

# Check available disk space (need at least 20GB)
available_space=$(df . | awk 'NR==2 {print $4}')
required_space=20000000  # 20GB in KB

if [ $available_space -lt $required_space ]; then
    echo "❌ Insufficient disk space. Need at least 20GB, but only $(($available_space / 1000000))GB available."
    exit 1
fi

echo "✅ Sufficient disk space available"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read -r gpu_name memory; do
        echo "   GPU: $gpu_name with ${memory}MB VRAM"
    done
else
    echo "⚠️  No NVIDIA GPU detected. Will use CPU (slower but functional)."
fi

# Create virtual environment
echo ""
echo "🏗️  Creating virtual environment..."
python3 -m venv fin-o1-env

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created successfully"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment and install dependencies
echo ""
echo "📦 Activating virtual environment and installing dependencies..."
source fin-o1-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source fin-o1-env/bin/activate"
echo "2. Download the model: python download_model.py"
echo "3. Run the web demo: python demo.py"
echo "4. Or run CLI demo: python demo.py --cli"
echo ""
echo "💡 Quick activation and run:"
echo "   source fin-o1-env/bin/activate && python download_model.py"
echo ""
echo "Note: Model download will take some time and requires ~16GB of data."
echo ""
echo "🔧 To deactivate the virtual environment later, run: deactivate"