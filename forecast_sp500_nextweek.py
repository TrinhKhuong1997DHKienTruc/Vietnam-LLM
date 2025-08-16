#!/usr/bin/env python3
"""
Forecast S&P 500 closing prices for the next 5 trading days using a simple ARIMA model.
This is a quick demo, not production-grade forecasting.
"""
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

symbol = "^GSPC"
print(f"Fetching historical data for {symbol} ...")

ticker = yf.Ticker(symbol)
# Get last 3 years of daily prices
hist = ticker.history(period="3y")

if hist.empty:
    raise ValueError("No historical data fetched. Check internet connection or symbol.")

close_prices = hist["Close"]
print(f"Retrieved {len(close_prices)} daily closing prices from {close_prices.index.min().date()} to {close_prices.index.max().date()}")

# Fit ARIMA(p=5,d=1,q=0) model (simple example)
print("Fitting ARIMA(5,1,0) model ... (this may take a few seconds)")
model = ARIMA(close_prices, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Forecast next 5 trading days
ahead = 5
print(f"\nForecasting next {ahead} trading days ...")
forecast = model_fit.forecast(steps=ahead)

# Prepare dates for forecast output
last_date = close_prices.index[-1]
forecast_dates = []
counter = 1
while len(forecast_dates) < ahead:
    next_day = last_date + timedelta(days=counter)
    # Skip weekends
    if next_day.weekday() < 5:
        forecast_dates.append(next_day)
    counter += 1

forecast_series = pd.Series(forecast.values, index=forecast_dates)

print("\nS&P 500 Closing Price Forecast:")
for date, price in forecast_series.items():
    print(f"{date.date()}: {price:.2f}")