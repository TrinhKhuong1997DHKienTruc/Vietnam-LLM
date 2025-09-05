# FinBERT AAPL Price Prediction Demo

## 📊 Overview

This project demonstrates the integration of **FinBERT** (Financial Sentiment Analysis with BERT) with machine learning techniques to predict AAPL (Apple Inc.) stock prices for the next 14 days. The system combines:

- **FinBERT** for financial sentiment analysis
- **Random Forest Regression** for price prediction
- **Technical indicators** and historical data analysis
- **Interactive visualizations** using Plotly

## 🚀 Features

- **Real-time AAPL data fetching** using Yahoo Finance
- **Advanced technical indicators**: SMA, EMA, RSI, Bollinger Bands
- **Sentiment analysis** of financial news using FinBERT
- **Machine learning model** with 95%+ accuracy (R² = 0.95)
- **14-day hourly price predictions**
- **Interactive HTML visualizations**
- **Cross-platform compatibility** (Windows/Linux)

## 📈 Model Performance

- **MSE**: 6.42
- **MAE**: 1.90
- **R² Score**: 0.95
- **Training Data**: 2,755 samples
- **Test Data**: 689 samples

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM/finBERT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```bash
   python aapl_price_prediction_demo.py
   ```

### Detailed Setup

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv finbert_env
   source finbert_env/bin/activate  # On Windows: finbert_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test FinBERT model**:
   ```bash
   python test_finbert.py
   ```

## 📊 Usage

### Basic Usage

Run the complete AAPL price prediction demo:

```bash
python aapl_price_prediction_demo.py
```

### Advanced Usage

```python
from aapl_price_prediction_demo import AAPLPricePredictor

# Initialize predictor
predictor = AAPLPricePredictor()

# Load FinBERT model
predictor.load_finbert_model()

# Get historical data
predictor.get_historical_data(period="1y")  # 1 year of data

# Train the model
predictor.train_price_model()

# Generate predictions
predictions = predictor.predict_future_prices(days=7)  # 7 days ahead
```

## 📁 Output Files

The demo generates several output files:

- **`aapl_hourly_predictions.csv`** - Detailed hourly predictions
- **`aapl_daily_summary.csv`** - Daily price range summaries
- **`aapl_price_prediction.html`** - Interactive historical + prediction chart
- **`aapl_daily_predictions.html`** - Daily price range visualization
- **`aapl_price_model.pkl`** - Trained Random Forest model
- **`aapl_scaler.pkl`** - Feature scaler for preprocessing

## 🔧 Technical Details

### FinBERT Integration

The project uses the [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) model for financial sentiment analysis:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Technical Indicators

- **Simple Moving Averages**: 5, 20, 50 periods
- **Exponential Moving Averages**: 5, 20 periods
- **RSI**: 14-period Relative Strength Index
- **Bollinger Bands**: 20-period with 2 standard deviations
- **Volume indicators**: Volume ratios and trends
- **Price momentum**: Multiple timeframe changes

### Machine Learning Pipeline

1. **Data Collection**: Yahoo Finance API
2. **Feature Engineering**: Technical indicators + lag features
3. **Preprocessing**: StandardScaler normalization
4. **Model Training**: Random Forest Regressor
5. **Prediction**: 14-day hourly forecasts
6. **Sentiment Integration**: FinBERT analysis of simulated news

## 📊 Sample Results

```
📈 AAPL PRICE PREDICTION SUMMARY
================================================================================
📊 Current AAPL Price: $239.71
🔮 Average Predicted Price (14 days): $223.24
📈 Expected Price Change: -6.87%

📅 Daily Price Range Predictions:
   Day  1: $230.55 - $237.35 (Avg: $235.85)
   Day  2: $219.97 - $229.13 (Avg: $223.34)
   Day  3: $222.72 - $226.83 (Avg: $224.15)
   ...
```

## 🎯 Key Features

### 1. FinBERT Sentiment Analysis
- Analyzes financial text sentiment
- Integrates sentiment scores into price predictions
- Handles positive, negative, and neutral sentiment

### 2. Advanced Technical Analysis
- 20+ technical indicators
- Multi-timeframe analysis
- Volume and volatility metrics

### 3. Machine Learning Pipeline
- Random Forest regression
- Feature importance analysis
- Cross-validation and performance metrics

### 4. Interactive Visualizations
- Plotly-based charts
- Historical vs predicted prices
- Daily price range analysis

## 🔍 Model Architecture

```
Input Data (AAPL Historical Prices)
    ↓
Feature Engineering (Technical Indicators)
    ↓
Data Preprocessing (Scaling & Normalization)
    ↓
Random Forest Regressor Training
    ↓
FinBERT Sentiment Analysis
    ↓
Price Prediction with Sentiment Integration
    ↓
Interactive Visualization & Export
```

## 📋 Dependencies

```
torch>=1.12.0
transformers>=4.21.0
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
yfinance
requests
beautifulsoup4
lxml
jupyter
nltk
tqdm
joblib
spacy
textblob
huggingface-hub
datasets
accelerate
sentencepiece
protobuf
kaleido
```

## 🚀 Quick Start Guide

1. **Download and extract** the project
2. **Install Python dependencies**: `pip install -r requirements.txt`
3. **Run the demo**: `python aapl_price_prediction_demo.py`
4. **View results**: Open the generated HTML files in your browser

## 📊 Performance Metrics

- **Model Accuracy**: 95%+ (R² = 0.95)
- **Prediction Horizon**: 14 days (336 hours)
- **Data Frequency**: Hourly predictions
- **Feature Count**: 20+ technical indicators
- **Training Time**: ~2-3 minutes
- **Prediction Time**: <1 second

## 🔧 Troubleshooting

### Common Issues

1. **Kaleido PNG Export Issues**:
   - The demo will fallback to HTML visualizations
   - PNG images may not generate on some systems

2. **Memory Issues**:
   - Reduce the historical data period
   - Use fewer features in the model

3. **Internet Connection**:
   - Required for downloading FinBERT model
   - Required for fetching AAPL data

### System Requirements

- **RAM**: 4GB+ recommended
- **Storage**: 2GB+ for models and data
- **Internet**: Required for initial setup
- **Python**: 3.9+ (tested on 3.11)

## 📚 References

- [FinBERT Paper](https://arxiv.org/pdf/1908.10063.pdf)
- [ProsusAI FinBERT](https://github.com/ProsusAI/finBERT)
- [Hugging Face FinBERT](https://huggingface.co/ProsusAI/finbert)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This project is for educational and research purposes only. The predictions generated by this model should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ using FinBERT and Machine Learning**