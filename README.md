# Fin-o1-14B Model Runner

This repository contains scripts to run the [Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B) large language model from Hugging Face.

## Model Information

- **Model**: Fin-o1-14B
- **Parameters**: 14 billion
- **Source**: [TheFinAI/Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B)
- **Type**: Causal Language Model
- **Domain**: Financial and general language understanding

## System Requirements

### Minimum Requirements
- **RAM**: 32GB+ (for CPU inference)
- **Storage**: 30GB+ free space
- **Python**: 3.8+

### Recommended Requirements (GPU)
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 30GB+ free space
- **Python**: 3.8+

## Installation

1. **Clone or navigate to this directory**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Simple Pipeline Runner (Recommended for beginners)

```bash
python simple_run.py
```

This script uses Hugging Face's pipeline API and is easier to use.

### Option 2: Advanced Runner with Quantization

```bash
python run_model.py
```

This script includes:
- 4-bit quantization for memory efficiency
- Better memory management
- More configuration options

## Features

- **Interactive Mode**: Chat with the model in real-time
- **Memory Optimization**: Automatic quantization and memory management
- **Device Detection**: Automatically detects and uses GPU if available
- **Error Handling**: Robust error handling and recovery

## Example Usage

Once the model is loaded, you can interact with it:

```
You: What are the main factors that affect stock prices?
Assistant: Stock prices are influenced by several key factors including company earnings, 
economic indicators, market sentiment, interest rates, and geopolitical events...

You: Can you explain quantitative easing?
Assistant: Quantitative easing (QE) is a monetary policy tool used by central banks...
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Ensure you have sufficient RAM/VRAM
   - The model requires significant memory resources
   - Consider using the quantized version in `run_model.py`

2. **Model Download Issues**
   - Check your internet connection
   - Ensure you have sufficient disk space
   - The model is ~30GB and may take time to download

3. **Slow Performance on CPU**
   - This is expected for a 14B parameter model
   - Consider using a GPU for better performance
   - The model will work on CPU but will be significantly slower

### Performance Tips

- **GPU Users**: Ensure CUDA is properly installed
- **Memory Management**: Close other applications to free up memory
- **Batch Size**: Keep input lengths reasonable for better performance

## Model Capabilities

The Fin-o1-14B model is trained on financial data and can:
- Answer questions about financial markets
- Explain economic concepts
- Provide investment insights
- Handle general language tasks
- Generate human-like text responses

## License

This project is for educational and research purposes. Please refer to the original model's license on Hugging Face.

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify you have sufficient system resources
4. Check the Hugging Face model page for updates
