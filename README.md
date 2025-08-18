# Fin-o1-8B Model Demo

🚀 **A ready-to-use implementation of the Fin-o1-8B language model from Hugging Face**

This project provides a complete setup for running the Fin-o1-8B model, a specialized language model trained on financial data and general knowledge. The model is optimized for financial analysis, investment advice, and general Q&A tasks.

## 🌟 Features

- **Pre-configured environment** with all dependencies
- **Interactive chat interface** for real-time conversations
- **Demo examples** showcasing financial knowledge
- **Cross-platform compatibility** (Windows, Linux, macOS)
- **Memory-efficient loading** with automatic device detection
- **Simple setup** - just download and run!

## 📋 Requirements

- **Python**: 3.8+ (Tested on Python 3.13)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: At least 20GB free space for model download
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

## 🚀 Quick Start

### Option 1: Download and Run (Recommended)

1. **Download the project**
   ```bash
   # Clone the repository
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM
   
   # Or download ZIP from GitHub and extract
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv fin_o1_env
   
   # Activate virtual environment
   # On Windows:
   fin_o1_env\Scripts\activate
   # On Linux/macOS:
   source fin_o1_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the demo**
   ```bash
   # Simple demo (recommended for first run)
   python fin_o1_simple.py
   
   # Interactive chat interface
   python fin_o1_demo.py
   ```

### Option 2: One-Command Setup (Linux/macOS)

```bash
# Download and setup in one command
curl -sSL https://raw.githubusercontent.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM/main/setup.sh | bash
```

## 📁 Project Structure

```
Vietnam-LLM/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── fin_o1_simple.py         # Simple demo script
├── fin_o1_demo.py           # Interactive chat script
├── setup.sh                  # Automated setup script (Linux/macOS)
├── setup.bat                 # Automated setup script (Windows)
└── .gitignore               # Git ignore file
```

## 🎯 Usage Examples

### Running the Simple Demo

```bash
python fin_o1_simple.py
```

This will:
- Download the Fin-o1-8B model (first run only)
- Run predefined financial questions
- Show model responses

### Interactive Chat Mode

```bash
python fin_o1_demo.py
```

This provides:
- Interactive chat interface
- Real-time responses
- Menu-driven options

### Sample Questions You Can Ask

- "What is the difference between stocks and bonds?"
- "Explain compound interest in simple terms"
- "What is diversification in investing?"
- "How does inflation affect investments?"
- "What are the main types of investment portfolios?"

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Close other applications
   - Use the simple demo first: `python fin_o1_simple.py`
   - Ensure you have at least 8GB RAM available

2. **Model Download Issues**
   - Check internet connection
   - Ensure sufficient disk space (20GB+)
   - Try running again - the model will resume download

3. **Python Version Issues**
   - Ensure Python 3.8+ is installed
   - Use virtual environment to avoid conflicts

4. **CUDA/GPU Issues**
   - The model works on CPU if GPU is not available
   - For GPU acceleration, ensure CUDA is properly installed

### Performance Tips

- **First run**: Model download takes 5-10 minutes
- **Subsequent runs**: Model loads in 1-2 minutes
- **Memory usage**: ~8-12GB RAM during operation
- **Response time**: 2-10 seconds depending on question complexity

## 🌍 Cross-Platform Support

### Windows
- Tested on Windows 10/11
- Use `setup.bat` for automated setup
- Ensure Python 3.8+ is installed

### Linux
- Tested on Ubuntu 18.04+, CentOS 7+
- Use `setup.sh` for automated setup
- Works with most Python distributions

### macOS
- Tested on macOS 10.15+
- Use `setup.sh` for automated setup
- Compatible with Homebrew Python

## 📚 Model Information

**Fin-o1-8B** is a specialized language model trained on:
- Financial data and analysis
- Investment strategies
- Economic concepts
- General knowledge
- Professional writing

**Model Specifications:**
- **Size**: 8 billion parameters
- **Architecture**: Transformer-based
- **Training Data**: Financial and general knowledge corpus
- **Specialization**: Financial analysis and Q&A

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TheFinAI** for providing the Fin-o1-8B model
- **Hugging Face** for the transformers library
- **PyTorch** team for the deep learning framework

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

## 🔄 Updates

This project is actively maintained. Check back regularly for:
- Performance improvements
- New features
- Bug fixes
- Compatibility updates

---

**Happy exploring with Fin-o1-8B! 🚀**

*Built with ❤️ for the AI community*
