#!/bin/bash

echo "🦙 Fin-o1-8B Setup Script"
echo "=========================="

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

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 detected"

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

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Download the model: python3 download_model.py"
echo "2. Run the web demo: python3 demo.py"
echo "3. Or run CLI demo: python3 demo.py --cli"
echo ""
echo "Note: Model download will take some time and requires ~16GB of data."