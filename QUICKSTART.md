# 🚀 Quick Start Guide - Fin-o1-8B Demo

## ⚡ Immediate Start

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Run Command Line Demo (Recommended First)
```bash
python fin_o1_demo.py
```
This will:
- Download the Fin-o1-8B model (~16GB)
- Run predefined financial examples
- Start interactive mode

### 3. Run Web Interfaces

**Streamlit (Full-featured):**
```bash
streamlit run streamlit_app.py
```
- Opens at http://localhost:8501
- Chat interface, examples, analytics dashboard

**Gradio (Simple):**
```bash
python gradio_app.py
```
- Opens at http://localhost:7860
- Clean, simple interface

## 🎯 What You'll Get

- **Financial Reasoning**: Math, economics, business analysis
- **Interactive Chat**: Ask any financial question
- **Pre-built Examples**: ROI, profit margins, compound interest
- **Multiple Interfaces**: CLI, Streamlit, Gradio

## ⚠️ Important Notes

- **First Run**: Model download takes 10-30 minutes
- **Hardware**: Works on CPU (slower) or GPU (faster)
- **Memory**: Requires ~16GB RAM minimum
- **Storage**: ~20GB free space needed

## 🔧 Troubleshooting

- **Out of Memory**: Close other applications
- **Slow Performance**: Use GPU if available
- **Download Issues**: Check internet connection

## 📚 Full Documentation

See `README.md` for complete details and examples.

---

**Ready to start? Run: `source venv/bin/activate && python fin_o1_demo.py`**