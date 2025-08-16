# 🦙 Fin-o1-8B Financial Reasoning Model Demo

This repository contains a complete setup for running the **Fin-o1-8B** model locally. Fin-o1-8B is a fine-tuned version of Qwen3-8B, specifically designed to excel at financial reasoning tasks.

## 📊 Model Information

- **Model**: [TheFinAI/Fin-o1-8B](https://huggingface.co/TheFinAI/Fin-o1-8B)
- **Base Model**: Qwen3-8B
- **Parameters**: 8.19B
- **Specialization**: Financial reasoning and mathematical tasks
- **License**: Apache 2.0
- **Paper**: [Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance](https://arxiv.org/abs/2502.08127)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Model

```bash
python download_model.py
```

**Note**: This will download approximately 16GB of model files. Ensure you have sufficient disk space and a stable internet connection.

### 3. Run the Demo

#### Web Interface (Recommended)
```bash
python demo.py
```
Then open your browser to `http://localhost:7860`

#### Command Line Interface
```bash
python demo.py --cli
```

## 🏗️ System Requirements

### Minimum Requirements
- **RAM**: 16GB+ (32GB+ recommended)
- **Storage**: 20GB+ free space
- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)

### Recommended Requirements
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (A100, V100, RTX 3090, etc.)
- **Storage**: SSD with 50GB+ free space

## 📁 Project Structure

```
.
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── download_model.py      # Model download script
├── demo.py               # Main demo application
└── fin-o1-8b/           # Downloaded model files (created after download)
```

## 🔧 Usage Examples

### Financial Calculations
- Profit margin calculations
- Compound interest computations
- Stock price percentage changes
- Financial ratio analysis

### Mathematical Reasoning
- Basic arithmetic operations
- Complex mathematical problems
- Step-by-step problem solving

### Example Questions
1. "What is the result of 3-5?"
2. "If a company has revenue of $1M and costs of $600K, what is the profit margin?"
3. "Calculate the compound interest on $1000 at 5% for 3 years."
4. "What is the difference between simple and compound interest?"
5. "If a stock price increases from $50 to $75, what is the percentage gain?"

## 🎛️ Configuration Options

### Generation Parameters
- **Max New Tokens**: Control response length (64-1024)
- **Temperature**: Control creativity/randomness (0.1-1.5)

### Model Loading
- Automatic device detection (CPU/GPU)
- Memory-efficient loading with `device_map="auto"`
- Support for both CUDA and CPU inference

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory Errors
- Reduce `max_length` parameter
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`
- Close other applications to free up RAM

#### Model Download Failures
- Check internet connection
- Ensure sufficient disk space
- Try running `python download_model.py` again

#### Import Errors
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility

### Performance Tips
- Use GPU acceleration when available
- Adjust temperature for more focused responses
- Use appropriate max_length for your use case

## 🔬 Research and Citation

If you use this model in your research, please cite:

```bibtex
@article{qian2025fino1,
  title={Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance},
  author={Qian, Lingfei and Zhou, Weipeng and Wang, Yan and Peng, Xueqing and Huang, Jimin and Xie, Qianqian},
  journal={arXiv preprint arXiv:2502.08127},
  year={2025}
}
```

## 📚 Additional Resources

- [Model Card on Hugging Face](https://huggingface.co/TheFinAI/Fin-o1-8B)
- [Research Paper](https://arxiv.org/abs/2502.08127)
- [TheFinAI Organization](https://huggingface.co/TheFinAI)
- [Base Model: Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this demo.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

**Note**: The Fin-o1-8B model is designed for research and educational purposes. Please ensure compliance with the model's license terms when using it in production applications.
