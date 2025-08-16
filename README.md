# Fin-o1-8B Model Runner

This project provides a complete setup to run the [Fin-o1-8B model](https://huggingface.co/TheFinAI/Fin-o1-8B), a fine-tuned version of Qwen3-8B specifically designed for financial reasoning tasks.

## 🚀 Model Overview

**Fin-o1-8B** is a specialized language model that excels at:
- Financial calculations and analysis
- Investment reasoning
- Portfolio management concepts
- Economic logic and business benchmarks
- Financial document understanding

The model is based on Qwen3-8B and has been fine-tuned using SFT and GRPO on financial datasets including FinQA, TATQA, DocMath-Eval, Econ-Logic, BizBench-QA, and DocFinQA.

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- At least 16GB RAM (32GB+ recommended)
- 20GB+ free disk space for model download

## 🛠️ Installation

1. **Clone and navigate to the project directory:**
   ```bash
   cd /workspace
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
   ```

## 🚀 Quick Start

### Option 1: Interactive Chat (Recommended for first run)
```bash
python run_fin_o1.py --interactive
```

### Option 2: Single Prompt
```bash
python run_fin_o1.py --prompt "What is the compound interest on $10,000 at 7% for 10 years?"
```

### Option 3: Financial Demo
```bash
python financial_demo.py
```

### Option 4: Quick Test
```bash
python financial_demo.py --quick-test
```

## 🔧 Advanced Usage

### Command Line Options

```bash
python run_fin_o1.py [OPTIONS]

Options:
  --model TEXT           Model name or path (default: TheFinAI/Fin-o1-8B)
  --device TEXT          Device to run on: auto, cuda, cpu (default: auto)
  --8bit                 Load model in 8-bit precision (saves memory)
  --4bit                 Load model in 4-bit precision (saves more memory)
  --prompt TEXT          Single prompt to run
  --interactive          Start interactive chat session
  --max-tokens INTEGER   Maximum new tokens to generate (default: 512)
```

### Memory Optimization

For systems with limited GPU memory, use quantization:

```bash
# 8-bit quantization (recommended for 16GB+ VRAM)
python run_fin_o1.py --eight-bit --interactive

# 4-bit quantization (recommended for 8GB+ VRAM)
python run_fin_o1.py --four-bit --interactive
```

### Device Selection

```bash
# Auto-detect (recommended)
python run_fin_o1.py --device auto

# Force CPU usage
python run_fin_o1.py --device cpu

# Force CUDA usage
python run_fin_o1.py --device cuda
```

## 💡 Example Prompts

### Financial Calculations
- "Calculate the net present value of a $1000 investment with 5% annual return over 10 years"
- "What is the Sharpe ratio if expected return is 12%, risk-free rate is 3%, and standard deviation is 15%?"

### Investment Analysis
- "Compare the advantages and disadvantages of investing in stocks vs bonds"
- "Explain the concept of dollar-cost averaging and when it's most effective"

### Business Analysis
- "Analyze the financial health of a company with current ratio 2.5, debt-to-equity 0.6, and ROE 15%"
- "What factors should be considered when valuing a startup company?"

## 🏗️ Project Structure

```
.
├── run_fin_o1.py          # Main model runner script
├── financial_demo.py       # Financial reasoning demo
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Use `--4bit` flag for 4-bit quantization
   - Use `--8bit` flag for 8-bit quantization
   - Ensure sufficient RAM/VRAM

2. **Model Download Issues**
   - Check internet connection
   - Verify Hugging Face access
   - Clear Hugging Face cache: `huggingface-cli delete-cache`

3. **CUDA Issues**
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA compatibility
   - Use `--device cpu` as fallback

### Performance Tips

- **GPU Memory**: Use quantization for large models
- **Response Quality**: Lower temperature (0.3-0.7) for focused responses
- **Speed**: Higher temperature (0.8-1.2) for creative responses
- **Context**: Keep conversation history manageable

## 📊 Model Specifications

- **Base Model**: Qwen3-8B
- **Parameters**: 8.19B
- **Training Method**: SFT + GRPO
- **Specialization**: Financial reasoning
- **License**: Apache 2.0

## 🔬 Technical Details

The model uses:
- **Tokenizer**: Inherited from Qwen3-8B
- **Architecture**: Transformer-based causal language model
- **Precision**: BF16 (can be quantized to 8-bit or 4-bit)
- **Context Length**: Inherited from base model

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@article{qian2025fino1,
  title={Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance},
  author={Qian, Lingfei and Zhou, Weipeng and Wang, Yan and Peng, Xueqing and Huang, Jimin and Xie, Qianqian},
  journal={arXiv preprint arXiv:2502.08127},
  year={2025}
}
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [TheFinAI](https://huggingface.co/TheFinAI) for the Fin-o1-8B model
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [Hugging Face](https://huggingface.co/) for the transformers library
