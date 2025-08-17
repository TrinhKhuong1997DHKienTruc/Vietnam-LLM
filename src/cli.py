from __future__ import annotations

import argparse
import base64
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import load_settings
from src.data import fetch_ohlcv_daily, build_datasets
from src.model import build_lstm, train, forecast_next_week
from src.report import plot_forecast, render_html


def generate_report_for_ticker(ticker: str) -> str:
	settings = load_settings()
	df = fetch_ohlcv_daily(ticker)
	prepared = build_datasets(df, lookback=60)
	model = build_lstm((prepared.x_train.shape[1], prepared.x_train.shape[2]))
	result = train(model, prepared.x_train, prepared.y_train, prepared.x_val, prepared.y_val, epochs=25, batch_size=64)
	# forecast using last window from full series
	scaled_full = prepared.scaler.transform(df[["open", "high", "low", "close", "volume"]].astype(float).values)
	last_window = scaled_full[-60:, :]
	scaled_preds = forecast_next_week(result.model, last_window, steps=5)
	# Inverse scale: map predicted close (index 3)
	inv_preds = []
	for yhat in scaled_preds:
		row = last_window[-1].copy()
		row[3] = yhat
		inv = prepared.scaler.inverse_transform(row.reshape(1, -1))[0, 3]
		inv_preds.append(float(inv))
	# dates for last historical closes and forecast horizon (weekdays)
	last_date = df.index[-1]
	forecast_dates = []
	d = last_date
	while len(forecast_dates) < 5:
		d += pd.tseries.offsets.BDay(1)
		forecast_dates.append(pd.Timestamp(d))
	image_bytes = plot_forecast(list(df.index[-120:]), list(df["close"].values[-120:]), forecast_dates, inv_preds)
	plot_b64 = base64.b64encode(image_bytes).decode("ascii")
	html = render_html(
		ticker=ticker,
		generated_at=datetime.utcnow().isoformat(),
		last_close=float(df["close"].iloc[-1]),
		forecast_values=inv_preds,
		plot_base64=plot_b64,
	)
	report_path = os.path.join(settings.reports_dir, f"forecast_{ticker}.html")
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(html)
	return report_path


def main():
	parser = argparse.ArgumentParser(description="Train LSTM and forecast next week for one or more tickers.")
	parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers. Default from env or MSFT,NVDA")
	args = parser.parse_args()
	settings = load_settings()
	tickers = args.tickers.split(",") if args.tickers else settings.ticker_symbols
	report_paths = []
	for t in tickers:
		print(f"Processing {t}...")
		path = generate_report_for_ticker(t.strip().upper())
		print(f"Saved report: {path}")
		report_paths.append(path)
	print("Done.")


if __name__ == "__main__":
	main()