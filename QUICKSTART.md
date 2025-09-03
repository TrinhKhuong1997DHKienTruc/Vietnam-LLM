# 🚀 FinRobot Quick Start Guide

Get FinRobot up and running in minutes!

## ⚡ Quick Setup (Choose Your OS)

### Windows Users
```bash
# Double-click setup.bat or run in Command Prompt:
setup.bat
```

### Linux/macOS Users
```bash
# Make script executable and run:
chmod +x setup.sh
./setup.sh
```

### Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv finrobot_env

# 2. Activate environment
# Windows:
finrobot_env\Scripts\activate
# Linux/macOS:
source finrobot_env/bin/activate

# 3. Install dependencies
pip install -r requirements_simple.txt
```

## 🔑 Configure API Keys

1. **Copy sample files:**
   ```bash
   cp config_api_keys_sample config_api_keys
   cp OAI_CONFIG_LIST_sample OAI_CONFIG_LIST
   ```

2. **Edit config_api_keys with your keys:**
   - Get free Finnhub key: [finnhub.io](https://finnhub.io/)
   - Get free FMP key: [financialmodelingprep.com](https://financialmodelingprep.com/)
   - Get free SEC key: [sec.gov](https://sec.gov/)

3. **Edit OAI_CONFIG_LIST with OpenAI key:**
   - Get OpenAI key: [platform.openai.com](https://platform.openai.com/)

## 🧪 Test Installation

```bash
# Run simple demo
python simple_demo.py

# Generate comprehensive reports
python generate_reports.py
```

## 📊 What You'll Get

- **Real-time stock data** for MSFT and NVDA
- **Financial metrics** and ratios
- **Market sentiment analysis**
- **Investment recommendations**
- **Professional reports** in text format

## 🆘 Need Help?

- Check the main [README.md](README.md)
- Create an issue on GitHub
- Review the generated reports in the `reports/` directory

---

**Happy analyzing! 📈**