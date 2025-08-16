import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import os


def forecast_next_week(ticker: str, period_years: int = 5, forecast_days: int = 5):
    # Download historical data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * period_years)
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}.")

    # Use close price for modeling
    ts = df['Close'].dropna()

    # Fit ARIMA model automatically choosing order via AIC (simple grid search)
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    model = ARIMA(ts, order=(p,d,q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p,d,q)
                        best_model = model
                except Exception:
                    continue
    if best_model is None:
        raise RuntimeError("Unable to fit ARIMA model.")

    forecast = best_model.forecast(steps=forecast_days)
    last_date = ts.index[-1]
    forecast_index = pd.bdate_range(last_date + timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast.values, index=forecast_index, name='PredictedClose')

    return ts, forecast_series, best_order, best_aic


def save_report(ticker: str, ts: pd.Series, forecast: pd.Series, order, aic):
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    filename = os.path.join(report_dir, f"forecast_{ticker}_{datetime.today().strftime('%Y%m%d')}.md")

    with open(filename, 'w') as f:
        f.write(f"# {ticker} Price Forecast Report\n\n")
        f.write(f"Generated on {datetime.today().strftime('%Y-%m-%d')}\n\n")
        f.write("## Model Summary\n")
        f.write(f"* ARIMA Order: {order}\n")
        f.write(f"* AIC: {aic:.2f}\n\n")

        f.write("## Historical Closing Prices (last 10 entries)\n\n")
        f.write(ts.tail(10).to_markdown())
        f.write("\n\n")

        f.write("## Forecast for Next Week (5 Business Days)\n\n")
        f.write(forecast.to_markdown())
        f.write("\n\n")

    print(f"Report saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate price forecast using ARIMA")
    parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g., MSFT)')
    args = parser.parse_args()

    ts, forecast, order, aic = forecast_next_week(args.ticker)
    save_report(args.ticker, ts, forecast, order, aic)


if __name__ == '__main__':
    main()