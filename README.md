# NVDA Stock Price Prediction using Amazon Chronos-T5 Large Model

This project demonstrates how to predict NVIDIA (NVDA) stock prices for the next 14 days using Amazon's Chronos-T5 Large model, a state-of-the-art time series forecasting model.

## 🚀 Features

- **Full Chronos-T5 Large Model**: Uses the most powerful Chronos model for accurate predictions
- **Automatic Data Download**: Fetches real-time NVDA stock data from Yahoo Finance
- **Comprehensive Evaluation**: Provides detailed performance metrics (MAE, MSE, RMSE, MAPE)
- **Visual Reports**: Generates high-quality prediction plots with confidence intervals
- **Export Capabilities**: Saves predictions in CSV format for further analysis
- **Cross-Platform**: Works on both Windows and Linux systems

## 📊 Model Performance

The Chronos-T5 Large model achieved excellent performance on NVDA stock prediction:
- **Mean Absolute Error (MAE)**: $8.13
- **Root Mean Squared Error (RMSE)**: $9.75
- **Mean Absolute Percentage Error (MAPE)**: 4.69%

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Prediction**
   ```bash
   python chronos_nvda_prediction.py
   ```

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install scikit-learn>=1.0.0
pip install yfinance>=0.2.0
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

## 📁 Project Structure

```
Vietnam-LLM/
├── chronos_nvda_prediction.py    # Main prediction script
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── demo_reports/                # Generated reports folder
│   ├── NVDA_Prediction_Plot.png # Prediction visualization
│   ├── Evaluation_Report.txt    # Performance metrics
│   └── NVDA_Predictions.csv     # Predicted prices
└── NVDA.csv                     # Downloaded stock data
```

## 🔧 Usage

### Basic Usage

Simply run the main script:

```bash
python chronos_nvda_prediction.py
```

This will:
1. Download the latest NVDA stock data (500 days of historical data)
2. Load the Chronos-T5 Large model
3. Generate 14-day predictions
4. Create visualizations and reports
5. Save all outputs to the `demo_reports` folder

### Customizing Predictions

You can modify the prediction period by editing the `prediction_length` parameter in the script:

```python
# Change from 14 days to any number you prefer
prediction_length = 30  # Predict next 30 days
```

## 📈 Output Files

### 1. NVDA_Prediction_Plot.png
- High-resolution visualization showing:
  - Historical stock prices
  - Predicted future prices
  - 90% confidence intervals
  - Clear legends and professional formatting

### 2. Evaluation_Report.txt
- Detailed performance metrics
- Day-by-day price predictions
- Model information and timestamps
- Summary statistics

### 3. NVDA_Predictions.csv
- Machine-readable format
- Date and predicted price columns
- Easy to import into other analysis tools

## 🧠 Model Details

### Chronos-T5 Large Model
- **Architecture**: Transformer-based time series model
- **Parameters**: ~1.3 billion parameters
- **Training Data**: Diverse time series datasets
- **Capabilities**: 
  - Multi-horizon forecasting
  - Uncertainty quantification
  - Zero-shot forecasting on new domains

### Model Fallback Strategy
The script includes intelligent fallback mechanisms:
1. **Primary**: Chronos-T5 Large (most accurate)
2. **Fallback 1**: Chronos-T5 Base
3. **Fallback 2**: Chronos-T5 Small
4. **Fallback 3**: Chronos-T5 Tiny (fastest)

## 🔍 Technical Details

### Data Processing
- **Source**: Yahoo Finance API via yfinance
- **Period**: 500 days of historical data
- **Features**: Close prices (most stable for prediction)
- **Preprocessing**: Automatic normalization and tensor conversion

### Prediction Method
- **Context Length**: Uses all available historical data
- **Forecast Horizon**: 14 days (configurable)
- **Quantiles**: Provides 10th, 50th, and 90th percentiles
- **Uncertainty**: Confidence intervals for risk assessment

## 🌍 Cross-Platform Compatibility

### Windows
```bash
# Using Command Prompt
python chronos_nvda_prediction.py

# Using PowerShell
python chronos_nvda_prediction.py
```

### Linux/macOS
```bash
# Using bash
python3 chronos_nvda_prediction.py

# Or with python
python chronos_nvda_prediction.py
```

## 📊 Sample Results

Based on recent NVDA stock data, the model predicts:

| Day | Predicted Price | Confidence Range |
|-----|----------------|------------------|
| 1   | $178.38        | $171.34 - $187.77|
| 7   | $176.82        | $171.34 - $187.77|
| 14  | $183.86        | $171.34 - $187.77|

**Average Predicted Price**: $180.67
**Price Range**: $171.34 - $187.77

## ⚠️ Important Notes

### Disclaimer
- This is for educational and research purposes only
- Stock predictions are inherently uncertain
- Past performance does not guarantee future results
- Always consult with financial advisors before making investment decisions

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for large model)
- **Storage**: ~3GB for model download
- **Internet**: Required for data download and model loading

### Performance Tips
- Use GPU if available for faster inference
- Close other applications to free up memory
- Ensure stable internet connection for model download

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Amazon Science** for the Chronos forecasting models
- **Hugging Face** for model hosting and distribution
- **Yahoo Finance** for providing free stock data
- **Python community** for excellent open-source libraries

## 📞 Support

If you encounter any issues:
1. Check the [Issues](https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM/issues) page
2. Create a new issue with detailed error information
3. Include your system specifications and Python version

---

**Made with ❤️ for the financial AI community**