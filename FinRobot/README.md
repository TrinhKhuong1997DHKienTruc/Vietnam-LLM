# FinRobot: AI-Powered Stock Market Analysis Platform

<div align="center">
<img align="center" width="30%" alt="FinRobot Logo" src="https://github.com/AI4Finance-Foundation/FinRobot/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

## 🚀 Overview

**FinRobot** is an advanced AI-powered platform for comprehensive stock market analysis and forecasting. Built on the foundation of the original FinRobot project, this enhanced version provides real-time market data analysis, sentiment assessment, and investment recommendations using cutting-edge AI technology.

## ✨ Features

- **Real-time Market Data**: Live stock quotes, financial metrics, and company profiles
- **AI-Powered Analysis**: Advanced sentiment analysis and trend prediction
- **Comprehensive Reports**: Detailed stock analysis reports with investment recommendations
- **Multi-API Integration**: Finnhub, Financial Modeling Prep (FMP), and SEC data sources
- **Easy-to-Use Interface**: Simple Python scripts for quick analysis
- **Professional Output**: Formatted reports suitable for investment decision-making

## 🛠️ Installation

### Prerequisites

- Python 3.13+ (tested on Python 3.13.3)
- pip package manager
- Internet connection for API access

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM/FinRobot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv finrobot_env
   source finrobot_env/bin/activate  # On Windows: finrobot_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_simple.txt
   ```

4. **Configure API keys**
   - Edit `config_api_keys` file with your API keys
   - Edit `OAI_CONFIG_LIST` file with your OpenAI configuration

## 🔑 API Configuration

### Required API Keys

1. **Finnhub API Key** - Get free key at [finnhub.io](https://finnhub.io/)
2. **Financial Modeling Prep API Key** - Get free key at [financialmodelingprep.com](https://financialmodelingprep.com/)
3. **SEC API Key** - Get free key at [sec.gov](https://sec.gov/)
4. **OpenAI API Key** - Get key at [platform.openai.com](https://platform.openai.com/)

### Configuration Files

**config_api_keys**
```json
{
    "FINNHUB_API_KEY": "your_finnhub_key_here",
    "FMP_API_KEY": "your_fmp_key_here",
    "SEC_API_KEY": "your_sec_key_here"
}
```

**OAI_CONFIG_LIST**
```json
[
    {
        "model": "gpt-5-mini-2025-08-07",
        "api_key": "your_openai_key_here"
    }
]
```

## 📊 Usage Examples

### 1. Quick Stock Analysis

Run a simple analysis for any stock:

```bash
python simple_demo.py
```

This will analyze MSFT and NVDA stocks with basic metrics.

### 2. Comprehensive Report Generation

Generate detailed analysis reports:

```bash
python generate_reports.py
```

This creates professional-grade analysis reports saved in the `reports/` directory.

### 3. Custom Stock Analysis

Modify the scripts to analyze any stock symbol by changing the `symbol` variable.

## 📈 Sample Output

The platform generates comprehensive reports including:

- **Company Profile**: Name, industry, market cap, country
- **Market Data**: Current price, daily change, high/low, volume
- **Financial Metrics**: PE ratio, PEG ratio, price-to-book, etc.
- **Financial Performance**: Revenue, net income, assets, debt
- **Market Sentiment**: Trend analysis and prediction
- **Recent News**: Latest company developments
- **Investment Recommendation**: Buy/Sell/Hold with reasoning

## 🏗️ Project Structure

```
FinRobot/
├── finrobot/                 # Core FinRobot modules
├── tutorials_beginner/       # Beginner tutorials
├── tutorials_advanced/       # Advanced tutorials
├── reports/                  # Generated analysis reports
├── configs/                  # Configuration files
├── requirements_simple.txt   # Simplified dependencies
├── simple_demo.py           # Basic stock analysis demo
├── generate_reports.py      # Comprehensive report generator
├── config_api_keys          # API keys configuration
├── OAI_CONFIG_LIST          # OpenAI configuration
└── README.md                # This file
```

## 🔧 Customization

### Adding New Data Sources

The platform is designed to be easily extensible. Add new data sources by:

1. Creating new functions in the data fetching modules
2. Integrating with additional financial APIs
3. Extending the report generation logic

### Modifying Analysis Logic

Customize the analysis algorithms by:

1. Adjusting trend calculation parameters
2. Adding new technical indicators
3. Implementing custom sentiment analysis

## 📋 Requirements

### Core Dependencies

- `pyautogen>=0.2.19` - AI agent framework
- `requests>=2.31.0` - HTTP requests
- `pandas>=2.1.0` - Data manipulation
- `numpy>=1.26.4` - Numerical computing
- `finnhub-python` - Finnhub API client
- `yfinance` - Yahoo Finance data
- `sec_api` - SEC filings API

### Optional Dependencies

- `matplotlib` - Charting and visualization
- `mplfinance` - Financial charts
- `backtrader` - Backtesting framework

## 🚨 Disclaimer

**Important**: This software is for educational and informational purposes only. The analysis and recommendations provided should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AI4Finance Foundation** - Original FinRobot project
- **FinGPT Team** - Inspiration and research foundation
- **OpenAI** - Advanced language models
- **Financial Data Providers** - Market data APIs

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation in the `tutorials_beginner/` directory
- Review the original FinRobot documentation

## 🔄 Updates

This project is actively maintained and updated with:

- Latest financial data APIs
- Improved AI analysis algorithms
- Enhanced report generation
- Better error handling and reliability

---

**Built with ❤️ for the financial AI community**


