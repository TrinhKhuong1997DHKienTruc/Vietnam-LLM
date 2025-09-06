# FinRobot AAPL Stock Prediction Demo

This repository contains a comprehensive demonstration of the FinRobot AI Agent Platform for financial analysis, specifically focused on Apple Inc. (AAPL) stock price prediction.

## 🚀 Features

- **Market Forecaster Agent**: Predicts stock price movements using advanced AI analysis
- **Technical Analysis**: RSI, MACD, Moving Averages, and trend analysis
- **Comprehensive Reporting**: Detailed analysis reports with charts and predictions
- **14-Day Predictions**: Short-term stock price forecasting
- **Multiple AI Models**: Support for GPT-5 and Gemini models
- **Real-time Data**: Live stock data from Yahoo Finance
- **Visual Charts**: Professional price and volume charts

## 📊 Demo Results

The demo successfully analyzed AAPL stock and provided the following prediction:

- **Current Price**: $239.69
- **Prediction**: BULLISH
- **Expected Price Change**: +2.50%
- **Target Price**: $245.68
- **Confidence Score**: 25/100

### Technical Analysis Summary:
- **RSI (14)**: 64.19 (Neutral zone)
- **MACD**: Bullish crossover detected
- **Price Trend**: Above 20-day SMA (Bullish)
- **Volume**: Low trading volume

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.11)
- Windows, macOS, or Linux

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM/FinRobot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   
   Edit `OAI_CONFIG_LIST`:
   ```json
   [
       {
           "model": "gpt-5-mini-2025-08-07",
           "api_key": "your-openai-api-key-here",
           "base_url": "your-api-base-url-here"
       },
       {
           "model": "gemini-2.5-pro",
           "api_key": "your-gemini-api-key-here"
       }
   ]
   ```
   
   Edit `config_api_keys`:
   ```json
   {
       "FINNHUB_API_KEY": "your-finnhub-api-key",
       "FMP_API_KEY": "your-fmp-api-key",
       "SEC_API_KEY": "your-sec-api-key"
   }
   ```

## 🚀 Usage

### Run the AAPL Prediction Demo

```bash
python minimal_aapl_demo.py
```

This will:
1. Fetch real-time AAPL stock data
2. Perform technical analysis
3. Generate 14-day price prediction
4. Create comprehensive reports and charts
5. Save results to `demo/` folder

### Output Files

The demo creates the following files in the `demo/` folder:

```
demo/
├── logs/                    # Error logs
├── outputs/                 # Generated charts
│   └── aapl_comprehensive_chart.png
└── reports/                 # Analysis reports
    ├── aapl_analysis_report_YYYYMMDD_HHMMSS.txt
    └── aapl_analysis_summary_YYYYMMDD_HHMMSS.json
```

## 📈 Analysis Components

### 1. Data Collection
- Real-time stock prices from Yahoo Finance
- Historical data (1 year)
- Company information and financial metrics
- Trading volume analysis

### 2. Technical Analysis
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD (Moving Average Convergence Divergence)**: Trend following indicator
- **Moving Averages**: 20-day and 50-day SMAs
- **Volume Analysis**: Trading volume trends
- **Price Trend Analysis**: Support/resistance levels

### 3. Prediction Algorithm
The prediction combines multiple technical indicators:
- RSI signals (oversold/overbought)
- MACD crossover analysis
- Price trend relative to moving averages
- Volume confirmation
- Confidence scoring system

### 4. Risk Assessment
- Market volatility analysis
- Volume confirmation
- Technical indicator reliability
- Trend strength evaluation

## 🔧 Configuration

### API Keys Required

1. **OpenAI API**: For GPT-5 model access
2. **Google AI**: For Gemini model access
3. **Finnhub**: For financial data (optional)
4. **Financial Modeling Prep**: For additional data (optional)
5. **SEC API**: For regulatory filings (optional)

### Model Configuration

The demo supports multiple AI models:
- **GPT-5 Mini**: Primary analysis model
- **Gemini 2.5 Pro**: Alternative analysis model

## 📊 Sample Output

### Text Report
```
AAPL STOCK ANALYSIS REPORT
==================================================
Generated on: 2025-09-07 00:16:30

COMPANY INFORMATION:
------------------------------
Company: Apple Inc.
Current Price: $239.69
Price Change: $-0.09 (-0.04%)
Volume: 54,837,300
Market Cap: $3,557,095,374,848
P/E Ratio: 36.37

TECHNICAL ANALYSIS:
------------------------------
RSI (14): 64.19
MACD: 3.7217
MACD Signal: 3.0375
20-Day SMA: $231.15
Price Trend: Bullish
Volume Trend: Low

14-DAY PREDICTION:
------------------------------
Prediction: BULLISH
Confidence Score: 25
Expected Price Change: 2.50%
Target Price: $245.68

Key Factors:
  • RSI in neutral zone
  • MACD bullish crossover - positive momentum
  • Price above 20-day SMA - bullish trend
  • Low trading volume - weak conviction
```

### JSON Summary
```json
{
  "timestamp": "20250907_001630",
  "analysis_date": "2025-09-07 00:16:30",
  "company": "Apple Inc.",
  "current_price": 239.69,
  "price_change": -0.09,
  "price_change_pct": -0.04,
  "volume": 54837300,
  "market_cap": 3557095374848,
  "pe_ratio": 36.37,
  "technical_analysis": {
    "rsi": 64.19,
    "macd": 3.7217,
    "macd_signal": 3.0375,
    "price_trend": "Bullish",
    "volume_trend": "Low",
    "sma_20": 231.15,
    "sma_50": null
  },
  "prediction": {
    "prediction": "BULLISH",
    "confidence": 25,
    "price_change_pct": 2.5,
    "target_price": 245.68,
    "factors": [
      "RSI in neutral zone",
      "MACD bullish crossover - positive momentum",
      "Price above 20-day SMA - bullish trend",
      "Low trading volume - weak conviction"
    ]
  }
}
```

## 🎯 Use Cases

1. **Individual Investors**: Get AI-powered stock analysis and predictions
2. **Financial Advisors**: Enhance client recommendations with technical analysis
3. **Traders**: Short-term trading signals and market insights
4. **Researchers**: Study AI applications in financial markets
5. **Students**: Learn about technical analysis and AI in finance

## ⚠️ Disclaimer

This demo is for educational and research purposes only. The predictions and analysis provided are not financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.

## 🔗 Related Projects

- [FinRobot Original Repository](https://github.com/AI4Finance-Foundation/FinRobot)
- [Vietnam-LLM Repository](https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM)

## 📝 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or support, please open an issue in the repository.

---

**Note**: This demo requires valid API keys to function. Make sure to configure your API keys before running the analysis.
