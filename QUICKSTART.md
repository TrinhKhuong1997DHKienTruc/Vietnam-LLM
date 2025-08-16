# 🚀 Quick Start Guide

Get Fin-o1-8B running in 3 simple steps!

## ⚡ Fast Setup

### Step 1: Run Setup Script (Creates Virtual Environment)
```bash
./setup_venv.sh
```

This will:
- ✅ Check your system requirements
- 🏗️ Create a Python virtual environment
- 📦 Install all Python dependencies
- 🔍 Detect your GPU/CPU setup

### Step 2: Activate Environment & Download Model
```bash
source fin-o1-env/bin/activate
python download_model.py
```

**⚠️ Important**: This downloads ~16GB of model files. Ensure you have:
- Stable internet connection
- At least 20GB free disk space
- Patience (download takes 10-30 minutes depending on speed)

### Step 3: Run the Demo
```bash
python demo.py
```

Then open your browser to: **http://localhost:7860**

## 🎯 Alternative: Command Line Demo
```bash
source fin-o1-env/bin/activate
python demo.py --cli
```

## 🧪 Test Your Installation
```bash
source fin-o1-env/bin/activate
python test_model.py
```

## 📱 What You'll Get

- **Web Interface**: Beautiful Gradio UI with chat interface
- **Financial Expertise**: Specialized in financial reasoning and math
- **Example Questions**: Pre-loaded with financial examples
- **Adjustable Parameters**: Control response length and creativity

## 🔧 Virtual Environment Management

- **Activate**: `source fin-o1-env/bin/activate`
- **Deactivate**: `deactivate`
- **Quick Run**: `source fin-o1-env/bin/activate && python demo.py`

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed instructions
- Ensure you have Python 3.8+ and python3-venv installed
- For GPU users: Install CUDA 11.8+ and PyTorch with CUDA support

## 🎉 Success!

Once running, try these example questions:
- "What is the result of 3-5?"
- "Calculate 15% of 200."
- "If revenue is $1M and costs are $600K, what's the profit margin?"

---

**Happy Financial Reasoning! 🦙💰**