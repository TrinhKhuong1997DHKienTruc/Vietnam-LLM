# 🚀 Quick Start Guide

Get Fin-o1-14B running in 3 simple steps!

## Step 1: Setup Environment
```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

## Step 2: Activate Environment
```bash
source fin_o1_env/bin/activate
```

## Step 3: Run the Model
```bash
# Interactive mode (chat with the model)
python run_fin_o1.py

# Or run a single prompt
python run_fin_o1.py --prompt "What is the difference between stocks and bonds?"

# Or see the demo
python demo.py
```

## 🎯 What You'll Get

- **Fin-o1-14B**: A 14-billion parameter AI model specialized in finance
- **Interactive Chat**: Ask financial questions and get detailed answers
- **Optimized Performance**: Uses quantization to run efficiently on your hardware
- **Financial Expertise**: Specialized knowledge in:
  - Investment strategies
  - Financial analysis
  - Risk management
  - Market understanding
  - Banking & finance

## ⚡ Quick Test

After setup, test if everything works:
```bash
python test_setup.py
```

## 🆘 Need Help?

- **Memory Issues**: Use `--4bit` flag for lower memory usage
- **Slow Performance**: Ensure you have a GPU with CUDA support
- **Setup Problems**: Check the main README.md for detailed troubleshooting

## 💡 Pro Tips

- **First Run**: Will download ~28GB model (one-time)
- **GPU Memory**: 16GB+ recommended, 8GB minimum with 4-bit quantization
- **Best Performance**: Use NVIDIA GPU with CUDA support
- **CPU Only**: Will work but be very slow (not recommended)

Ready to start? Run `./setup.sh` now! 🚀