# AAPL Hourly Price Prediction using Chronos-Bolt Base Model

This project demonstrates how to use the Chronos-Bolt base model to predict AAPL (Apple Inc.) hourly stock prices for the next 14 days. The model provides quantile forecasts with confidence intervals, making it suitable for both point predictions and risk assessment.

## 🚀 Features

- **Real-time Data**: Downloads live AAPL data from Yahoo Finance
- **Advanced AI Model**: Uses Chronos-Bolt Base (205M parameters) for time series forecasting
- **Confidence Intervals**: Provides quantile predictions (10th to 90th percentiles)
- **Comprehensive Visualizations**: Multiple charts showing predictions and confidence bands
- **Easy to Use**: One-click execution with automatic dependency installation
- **Cross-platform**: Works on Windows, Linux, and macOS

## 🤖 Model Information

- **Model**: Chronos-Bolt Base (205M parameters)
- **Architecture**: T5 encoder-decoder
- **Training Data**: Nearly 100 billion time series observations
- **Performance**: Up to 250x faster than original Chronos models
- **Memory Efficiency**: 20x more memory-efficient
- **Zero-shot**: No fine-tuning required for new time series

## 📋 Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Internet connection for data download and model loading
- Optional: CUDA-compatible GPU for faster inference

## 🛠️ Installation

### Option 1: Quick Start (Recommended)

1. **Download the project**:
   ```bash
   git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git
   cd Vietnam-LLM/chronos-bolt-aapl-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test installation**:
   ```bash
   python test_installation.py
   ```

4. **Run prediction**:
   ```bash
   python aapl_prediction.py
   ```

### Option 2: Using Jupyter Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open and run**: `aapl_prediction_demo.ipynb`

## 📊 Usage

### Basic Usage

```python
from aapl_prediction import main
main()
```

### Advanced Usage

```python
import pandas as pd
from aapl_prediction import download_aapl_data, prepare_model, make_predictions

# Download data
data = download_aapl_data(days_back=90)

# Load model
model = prepare_model()

# Make predictions
predictions, context = make_predictions(model, data, prediction_hours=336)
```

## 📈 Output Files

The script generates several output files:

1. **`aapl_prediction_results.png`**: Comprehensive visualization with 4 subplots
   - Historical data + predictions with confidence intervals
   - Quantile predictions over time
   - Price distribution at different time points
   - Summary statistics

2. **`aapl_predictions_14days.csv`**: Detailed prediction data
   - Timestamps for each prediction
   - Quantile predictions (q10, q20, q30, q40, q50, q60, q70, q80, q90)
   - Hour index from prediction start

3. **`aapl_historical_data.csv`**: Historical price data used for context

## 🔧 Configuration

You can modify the following parameters in `aapl_prediction.py`:

- `days_back`: Number of days of historical data to download (default: 90)
- `prediction_hours`: Number of hours to predict (default: 336 = 14 days)
- `context_length`: Hours of historical data used as context (default: 168 = 7 days)

## 📊 Understanding the Predictions

### Quantile Forecasts

The model provides 9 quantile predictions:
- **q10, q90**: 80% confidence interval
- **q20, q80**: 60% confidence interval  
- **q50**: Median prediction (most likely outcome)

### Confidence Intervals

- **80% Confidence**: 80% of actual prices expected to fall within q10-q90 range
- **60% Confidence**: 60% of actual prices expected to fall within q20-q80 range
- **Median (q50)**: Most likely price prediction

### Trading Signals

The predictions can be used to generate trading signals:
- **BUY**: When predicted price > current price + threshold
- **SELL**: When predicted price < current price - threshold
- **HOLD**: When predicted price is within threshold

## 🚨 Important Notes

### Disclaimer

⚠️ **This is for educational and research purposes only. Do not use for actual trading decisions without proper risk management and additional analysis.**

### Model Limitations

- Predictions are based on historical patterns and may not account for:
  - Breaking news or market events
  - Fundamental analysis factors
  - Market sentiment changes
  - External economic factors

### Performance Considerations

- **First run**: Model download may take 5-10 minutes
- **CPU inference**: 2-5 minutes for predictions
- **GPU inference**: 30 seconds - 2 minutes for predictions
- **Memory usage**: 2-4GB RAM during inference

## 🔍 Troubleshooting

### Common Issues

1. **Import Error for chronos-forecasting**:
   ```bash
   pip install chronos-forecasting
   ```

2. **CUDA out of memory**:
   - The script automatically falls back to CPU
   - Reduce `prediction_hours` if needed

3. **No data downloaded**:
   - Check internet connection
   - Verify Yahoo Finance is accessible
   - Try different `days_back` value

4. **Model loading fails**:
   - Ensure stable internet connection
   - Check available disk space (model is ~800MB)
   - Try running `test_installation.py`

### Getting Help

If you encounter issues:

1. Run `python test_installation.py` to diagnose problems
2. Check the error messages in the console output
3. Ensure all dependencies are installed correctly
4. Verify Python version compatibility (3.8+)

## 📚 Technical Details

### Model Architecture

Chronos-Bolt uses a T5 encoder-decoder architecture:
- **Encoder**: Processes historical time series context
- **Decoder**: Generates quantile forecasts directly
- **Patches**: Time series chunked into patches for processing
- **Direct Multi-step**: Generates all future steps in one pass

### Data Processing

1. **Download**: Hourly AAPL data from Yahoo Finance
2. **Clean**: Remove missing values and outliers
3. **Context**: Use last 168 hours (7 days) as context
4. **Predict**: Generate 336 hours (14 days) of forecasts
5. **Filter**: Remove weekend predictions (trading hours only)

### Performance Metrics

The model provides several accuracy metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Amazon Research**: For developing the Chronos-Bolt model
- **Hugging Face**: For hosting the model and providing the inference library
- **Yahoo Finance**: For providing free financial data
- **PyTorch Team**: For the deep learning framework

## 📞 Contact

- **Author**: Trinh Khuong
- **Email**: trinhkhuong1997@gmail.com
- **GitHub**: [TrinhKhuong1997DHKienTruc](https://github.com/TrinhKhuong1997DHKienTruc)

## 🔗 Related Links

- [Chronos-Bolt Model on Hugging Face](https://huggingface.co/amazon/chronos-bolt-base)
- [Chronos Paper](https://openreview.net/forum?id=gerNCVqqtR)
- [AutoGluon Time Series](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

---

**Happy Predicting! 🚀📈**
