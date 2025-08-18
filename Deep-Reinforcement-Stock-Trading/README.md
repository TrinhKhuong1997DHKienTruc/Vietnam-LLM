# Deep Reinforcement Learning Stock Trading System

A comprehensive deep reinforcement learning system for stock trading and forecasting, featuring multiple AI models (DQN, DDQN, DDPG) with real-time stock data analysis and portfolio optimization.

## 🚀 Features

- **Multiple AI Models**: DQN, DDQN, and DDPG reinforcement learning agents
- **Real-time Stock Data**: Integration with multiple financial data APIs
- **Portfolio Optimization**: Advanced portfolio management and risk assessment
- **Comprehensive Reporting**: Detailed performance metrics and visualizations
- **Multi-stock Support**: Analyze and forecast multiple stocks simultaneously
- **Python 3.14 Compatible**: Updated dependencies for modern Python versions

## 📊 Supported Stocks

- **MSFT** (Microsoft Corporation)
- **NVDA** (NVIDIA Corporation)
- **Custom stocks** via CSV data files

## 🏗️ Architecture

```
Deep-Reinforcement-Stock-Trading/
├── agents/                 # AI model implementations
│   ├── DQN.py            # Deep Q-Network agent
│   ├── DDQN.py           # Double Deep Q-Network agent
│   └── DDPG.py           # Deep Deterministic Policy Gradient agent
├── data/                  # Stock data files
├── logs/                  # Training and evaluation logs
├── reports/               # Generated forecasting reports
├── saved_models/          # Trained model checkpoints
├── visualizations/        # Performance charts and graphs
├── config.py              # Configuration and API keys
├── download_stock_data.py # Stock data downloader
├── forecast_stocks.py     # Main forecasting script
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── utils.py               # Utility functions
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.14 or higher
- pip package manager
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM/Deep-Reinforcement-Stock-Trading
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download stock data**
   ```bash
   python download_stock_data.py
   ```

4. **Run forecasting models**
   ```bash
   python forecast_stocks.py
   ```

## 📈 Usage

### Training Models

Train a specific model on stock data:

```bash
python train.py --model_name DQN --stock_name MSFT_2023-01-01_2024-12-31 --num_episode 20
```

### Evaluating Models

Evaluate a trained model:

```bash
python evaluate.py --model_to_load DQN_ep20 --stock_name MSFT_2023-01-01_2024-12-31
```

### Running Full Forecast

Generate comprehensive forecasts for MSFT and NVDA:

```bash
python forecast_stocks.py
```

## 🔧 Configuration

### API Keys

The system supports multiple financial data APIs. Configure them in `config.py`:

```python
FINANCIAL_APIS = {
    "FINNHUB_API_KEY": "your_finnhub_key",
    "FMP_API_KEY": "your_fmp_key", 
    "SEC_API_KEY": "your_sec_key"
}
```

### Trading Parameters

Adjust trading parameters in `config.py`:

```python
TRADING_CONFIG = {
    "default_initial_balance": 50000,
    "default_window_size": 10,
    "default_episodes": 10,
    "risk_free_rate": 0.0275
}
```

## 📊 Model Performance Metrics

The system provides comprehensive performance analysis:

- **Portfolio Value**: Total portfolio worth over time
- **Total Return**: Absolute profit/loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Return Rates**: Daily return percentages

## 📁 Output Files

### Reports Directory

- `MSFT_forecast_report.txt` - Detailed MSFT analysis
- `NVDA_forecast_report.txt` - Detailed NVDA analysis
- `MSFT_portfolio_performance.png` - MSFT performance chart
- `NVDA_portfolio_performance.png` - NVDA performance chart

### Logs Directory

- Training logs with detailed episode information
- Evaluation logs with performance metrics
- Error logs for debugging

## 🎯 Model Comparison

| Model | Type | Use Case | Pros | Cons |
|-------|------|----------|------|------|
| **DQN** | Value-based | Basic trading | Simple, stable | May overestimate Q-values |
| **DDQN** | Value-based | Advanced trading | Reduces overestimation | More complex training |
| **DDPG** | Policy-based | Continuous control | Handles continuous actions | Requires more tuning |

## 🔍 Data Sources

- **Yahoo Finance**: Real-time stock data via yfinance
- **CSV Files**: Historical data in standard OHLCV format
- **Custom APIs**: Integration with Finnhub, FMP, and SEC APIs

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT**: This system is for educational and research purposes only. 

- Past performance does not guarantee future results
- Stock trading involves substantial risk of loss
- Always consult with financial advisors before making investment decisions
- The authors are not responsible for any financial losses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original implementation by [Albert-Z-Guo](https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading)
- Financial data APIs: Yahoo Finance, Finnhub, FMP, SEC
- Deep learning frameworks: TensorFlow, Keras

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the logs directory for error details
- Review the configuration in `config.py`

## 🔄 Updates

- **v2.0**: Python 3.14 compatibility, enhanced reporting
- **v1.0**: Original implementation with basic models

---

**Happy Trading! 📈💰**
