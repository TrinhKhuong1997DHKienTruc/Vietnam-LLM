# Fin-o1-14B Model Runner

This repository contains everything you need to run the [Fin-o1-14B model](https://huggingface.co/TheFinAI/Fin-o1-14B) from Hugging Face. This is a 14-billion parameter language model specifically trained for financial applications.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **NVIDIA GPU with CUDA support** (recommended for reasonable performance)
- **At least 16GB GPU memory** (8GB with 4-bit quantization)
- **~28GB free disk space** for the model
- **Stable internet connection** for initial download

### Option 1: Automated Setup (Recommended)

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the automated setup
./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv fin_o1_env

# Activate virtual environment
source fin_o1_env/bin/activate

# Install PyTorch (with CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

## 🎯 Running the Model

### Interactive Mode

```bash
# Activate virtual environment
source fin_o1_env/bin/activate

# Run in interactive mode
python run_fin_o1.py
```

### Single Prompt Mode

```bash
# Activate virtual environment
source fin_o1_env/bin/activate

# Run with a single prompt
python run_fin_o1.py --prompt "What is the difference between stocks and bonds?"
```

### Command Line Options

```bash
python run_fin_o1.py [OPTIONS]

Options:
  --model TEXT           Model name or path (default: TheFinAI/Fin-o1-14B)
  --prompt TEXT          Single prompt to run
  --max-length INTEGER   Maximum generation length (default: 512)
  --temperature FLOAT    Generation temperature (default: 0.7)
  --top-p FLOAT         Top-p sampling (default: 0.9)
  --no-8bit             Disable 8-bit quantization
  --4bit                Use 4-bit quantization instead of 8-bit
  --help                Show help message
```

## 💾 Memory Optimization

### 8-bit Quantization (Default)
- **GPU Memory**: ~16GB
- **Performance**: Good
- **Quality**: High

### 4-bit Quantization
- **GPU Memory**: ~8GB
- **Performance**: Good
- **Quality**: Slightly lower but still high

### No Quantization
- **GPU Memory**: ~28GB
- **Performance**: Best
- **Quality**: Best

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Use 4-bit quantization: `python run_fin_o1.py --4bit`
   - Close other applications using GPU memory
   - Consider using a cloud GPU instance

2. **Slow Performance on CPU**
   - This is expected for a 14B parameter model
   - Consider using Google Colab or a cloud GPU service

3. **Download Issues**
   - Ensure stable internet connection
   - The model is ~28GB, so download may take time
   - Use `huggingface-cli` for resumable downloads

### Performance Tips

- **Use GPU**: CPU inference is extremely slow for 14B models
- **Batch Processing**: Process multiple prompts together when possible
- **Model Caching**: The model will be cached after first download
- **Memory Management**: Use quantization if you have limited GPU memory

## 📊 Model Information

- **Model**: Fin-o1-14B
- **Parameters**: 14 billion
- **Architecture**: Based on Llama 2
- **Training**: Specialized for financial applications
- **License**: Check Hugging Face model page for licensing details

## 🌐 Alternative Deployment Options

### Google Colab
- Free GPU access (with limitations)
- No local setup required
- Good for testing and development

### Cloud GPU Services
- **AWS**: g4dn.xlarge or larger
- **Google Cloud**: T4 or V100 instances
- **Azure**: NC-series instances

### Local Deployment
- **Docker**: Use NVIDIA Docker for GPU support
- **Kubernetes**: For production deployments
- **API Server**: Wrap the model in a REST API

## 📝 Example Prompts

### Financial Analysis
```
"What are the key factors to consider when analyzing a company's financial statements?"
```

### Investment Advice
```
"Explain the difference between value investing and growth investing strategies."
```

### Market Understanding
```
"What causes stock market volatility and how can investors prepare for it?"
```

### Risk Management
```
"Describe the main types of investment risk and how to mitigate them."
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this setup.

## 📄 License

This setup script is provided as-is. Please check the model's license on the [Hugging Face page](https://huggingface.co/TheFinAI/Fin-o1-14B) for usage restrictions.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed error information
4. Consider using the Hugging Face community forums
