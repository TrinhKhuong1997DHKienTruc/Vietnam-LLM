# 💰 Fin-o1-8B Financial AI Demo

A comprehensive demonstration of the **Fin-o1-8B** model, a fine-tuned version of Qwen3-8B designed for financial reasoning tasks.

## 🧠 Model Overview

**Fin-o1-8B** is a state-of-the-art language model specifically designed for financial reasoning tasks. It's built upon the foundation of Qwen3-8B and fine-tuned using advanced techniques to excel in financial mathematics, economic logic, and business analysis.

### Key Features
- **Base Model**: Qwen3-8B (8.19B parameters)
- **Training Method**: Supervised Fine-Tuning (SFT) + Reinforcement Learning (RF)
- **Training Data**: FinCoT dataset derived from FinQA, TATQA, DocMath-Eval, Econ-Logic, BizBench-QA, DocFinQA
- **Model Size**: 8.19B parameters
- **License**: Apache 2.0

### Research Paper
**"Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance"**
- **Authors**: Qian, Lingfei; Zhou, Weipeng; Wang, Yan; Peng, Xueqing; Huang, Jimin; Xie, Qianqian
- **arXiv**: [2502.08127](https://arxiv.org/abs/2502.08127)
- **Hugging Face**: [TheFinAI/Fin-o1-8B](https://huggingface.co/TheFinAI/Fin-o1-8B)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- At least 16GB RAM (32GB+ recommended)
- At least 20GB free disk space

### Installation

1. **Clone and navigate to the project directory:**
```bash
cd fin-o1-demo
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download and run the model:**
```bash
python fin_o1_demo.py
```

## 📱 Available Interfaces

### 1. Command Line Interface (`fin_o1_demo.py`)
A comprehensive Python script with predefined examples and interactive mode.

```bash
python fin_o1_demo.py
```

**Features:**
- Automatic model downloading and loading
- Predefined financial reasoning examples
- Interactive chat mode
- Performance timing
- Device auto-detection (CUDA/CPU/MPS)

### 2. Streamlit Web App (`streamlit_app.py`)
A modern, feature-rich web interface with multiple tabs and visualizations.

```bash
streamlit run streamlit_app.py
```

**Features:**
- Chat interface with conversation history
- Predefined examples with one-click execution
- Analytics dashboard with financial visualizations
- Model information and documentation
- Responsive design with custom styling

### 3. Gradio Web App (`gradio_app.py`)
A simple, clean web interface for quick interactions.

```bash
python gradio_app.py
```

**Features:**
- Simple chat interface
- Model loading status
- Parameter controls
- Quick examples execution
- Lightweight and fast

## 🎯 Use Cases

The Fin-o1-8B model excels at:

- **Financial Calculations**: Interest rates, ROI, profit margins, break-even analysis
- **Economic Reasoning**: Inflation, monetary policy, market dynamics
- **Business Analysis**: Company evaluation, financial health assessment
- **Mathematical Problem Solving**: Arithmetic, algebra, financial mathematics
- **Investment Analysis**: Portfolio management, risk assessment
- **Document Analysis**: Financial reports, earnings statements

## 📊 Example Prompts

### Basic Math
```
What is the result of 3-5?
```

### Financial Calculations
```
If I invest $1000 at 5% annual interest for 3 years, how much will I have?
```

### Business Analysis
```
A company has revenue of $1M and expenses of $800K. What is the profit margin percentage?
```

### Economic Concepts
```
Explain the relationship between inflation and interest rates.
```

### Investment Analysis
```
What are the key factors to consider when evaluating a company's financial health?
```

## ⚙️ Configuration

### Model Parameters
- **Max Tokens**: Control response length (100-1000)
- **Temperature**: Control creativity (0.1-1.5, lower = more focused)
- **Device**: Automatic detection of CUDA/CPU/MPS

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Multi-core CPU with 16GB+ RAM
- **Storage**: 20GB+ free space for model download

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size or use CPU mode
   - Close other applications to free memory
   - Use model quantization if available

2. **Slow Performance**
   - Ensure CUDA is properly installed
   - Check GPU memory usage
   - Consider using CPU mode for smaller models

3. **Model Download Issues**
   - Check internet connection
   - Verify Hugging Face access
   - Clear Hugging Face cache if needed

### Performance Tips

- **GPU Mode**: Significantly faster than CPU mode
- **Batch Processing**: Process multiple prompts together when possible
- **Model Caching**: Models are cached after first download
- **Parameter Tuning**: Adjust temperature and max_tokens for optimal results

## 📁 Project Structure

```
fin-o1-demo/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── fin_o1_demo.py           # Command line interface
├── streamlit_app.py         # Streamlit web application
├── gradio_app.py            # Gradio web interface
└── .gitignore               # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TheFinAI** for developing and releasing the Fin-o1-8B model
- **Hugging Face** for hosting the model and providing the transformers library
- **Qwen Team** for the base Qwen3-8B model
- **Open Source Community** for the various libraries and tools used

## 📞 Support

- **Model Issues**: Check the [Hugging Face model page](https://huggingface.co/TheFinAI/Fin-o1-8B)
- **Code Issues**: Open an issue in this repository
- **Research Questions**: Refer to the [research paper](https://arxiv.org/abs/2502.08127)

---

**Happy Financial Reasoning! 💰🧠**
