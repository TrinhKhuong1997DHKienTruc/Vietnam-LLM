# Fin-o1-14B Model Runner

This repository contains scripts to download and run the [Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B) model from Hugging Face.

## Model Information

- **Model**: Fin-o1-14B
- **Size**: 14 billion parameters
- **Source**: [TheFinAI/Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B)
- **Type**: Large Language Model for text generation

## System Requirements

### Minimum Requirements
- **RAM**: 32GB+ (64GB+ recommended)
- **Storage**: 50GB+ free space for model download
- **Python**: 3.8+

### Recommended Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 64GB+
- **Storage**: SSD with 100GB+ free space

## Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Run the Model

#### Option A: Simple Runner (Recommended for first-time users)
```bash
python simple_run.py
```

#### Option B: Full Runner (More features, better error handling)
```bash
python run_fin_o1_14b.py
```

## Manual Installation

If you prefer to install dependencies manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (choose appropriate version)
# For CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

## Usage

### Interactive Chat
Both scripts provide interactive chat interfaces where you can:
- Type prompts and get AI-generated responses
- Type 'quit' or 'exit' to close the session
- Use Ctrl+C to interrupt generation

### Example Prompts
- "Explain quantum computing in simple terms"
- "Write a short story about a robot learning to paint"
- "What are the benefits of renewable energy?"

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - The model requires significant GPU memory
   - Try using 4-bit quantization (already enabled in scripts)
   - Close other applications using GPU memory

2. **Slow Performance**
   - Ensure you're using a GPU with CUDA support
   - CPU-only mode will be extremely slow for a 14B model

3. **Model Download Issues**
   - Check internet connection
   - Ensure sufficient disk space
   - Try running the script again

4. **Dependency Issues**
   - Make sure you're in the virtual environment
   - Try reinstalling requirements: `pip install -r requirements.txt`

### Memory Optimization

The scripts automatically use:
- 4-bit quantization to reduce memory usage
- Mixed precision (float16) for faster inference
- Automatic device mapping for optimal GPU utilization

## Files Description

- `setup.sh` - Automated environment setup script
- `requirements.txt` - Python dependencies
- `simple_run.py` - Simple model runner using Hugging Face pipeline
- `run_fin_o1_14b.py` - Full-featured model runner with advanced options
- `README.md` - This documentation file

## Notes

- **First Run**: The initial run will download the 14B parameter model, which may take 30+ minutes depending on your internet speed
- **Model Size**: The downloaded model will be approximately 28GB (compressed)
- **GPU Memory**: Even with quantization, you'll need at least 16GB GPU memory for optimal performance

## License

This project is provided as-is for educational and research purposes. Please refer to the original model's license on Hugging Face for commercial usage terms.
