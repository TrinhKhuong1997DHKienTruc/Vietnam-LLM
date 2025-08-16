# 🛠️ Installation Guides

This document provides multiple installation approaches for different environments and system configurations.

## 🐍 Environment 1: Standard Python Installation

### Prerequisites
- Python 3.8+ installed
- pip available
- 20GB+ free disk space

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download model
python download_model.py

# Run demo
python demo.py
```

## 🏗️ Environment 2: Virtual Environment (Recommended)

### Prerequisites
- Python 3.8+ with venv module
- 20GB+ free disk space

### Installation
```bash
# Create virtual environment
python3 -m venv fin-o1-env

# Activate it
source fin-o1-env/bin/activate  # Linux/Mac
# or
fin-o1-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download model
python download_model.py

# Run demo
python demo.py

# Deactivate when done
deactivate
```

## 📦 Environment 3: Conda/Miniconda

### Prerequisites
- Conda or Miniconda installed
- 20GB+ free disk space

### Installation
```bash
# Create conda environment
conda create -n fin-o1 python=3.10

# Activate environment
conda activate fin-o1

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Download model
python download_model.py

# Run demo
python demo.py

# Deactivate when done
conda deactivate
```

## 🚀 Environment 4: Docker (Containerized)

### Prerequisites
- Docker installed
- 20GB+ free disk space

### Installation
```bash
# Build Docker image
docker build -t fin-o1-8b .

# Run container
docker run -it --rm -p 7860:7860 -v $(pwd)/fin-o1-8b:/app/fin-o1-8b fin-o1-8b

# Or run with GPU support (if available)
docker run -it --rm --gpus all -p 7860:7860 -v $(pwd)/fin-o1-8b:/app/fin-o1-8b fin-o1-8b
```

## 🔧 Environment 5: System Package Override (Use with Caution)

### Prerequisites
- Python 3.8+ installed
- pip available
- 20GB+ free disk space
- **Warning**: This may affect system stability

### Installation
```bash
# Install with system override (use with caution)
pip install --break-system-packages -r requirements.txt

# Download model
python download_model.py

# Run demo
python demo.py
```

## 🐳 Dockerfile

If you want to use Docker, here's the Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Default command
CMD ["python", "demo.py"]
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "externally-managed-environment" Error
**Solution**: Use virtual environment or Docker approach

#### 2. Out of Memory Errors
**Solutions**:
- Reduce `max_length` parameter in demo
- Use CPU instead of GPU
- Close other applications
- Increase system swap space

#### 3. CUDA/GPU Issues
**Solutions**:
- Install correct PyTorch version for your CUDA
- Check GPU compatibility
- Use CPU fallback: `export CUDA_VISIBLE_DEVICES=""`

#### 4. Import Errors
**Solutions**:
- Verify all dependencies are installed
- Check Python version compatibility
- Use virtual environment to avoid conflicts

### Performance Tips

#### For CPU Users
- Use smaller `max_length` values
- Reduce temperature for faster responses
- Close unnecessary applications

#### For GPU Users
- Ensure PyTorch is compiled with CUDA support
- Use appropriate batch sizes
- Monitor GPU memory usage

## 📋 System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| Storage | 20GB | 50GB+ |
| Python | 3.8+ | 3.10+ |
| CUDA | 11.8+ | 12.0+ |
| GPU VRAM | 8GB+ | 16GB+ |

## 🎯 Quick Test Commands

After installation, test with:

```bash
# Test model loading
python test_model.py

# Test web interface
python demo.py

# Test command line
python demo.py --cli
```

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

---

**Choose the installation method that best fits your environment and requirements!**