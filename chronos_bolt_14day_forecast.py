import os
import json
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
	import yfinance as yf
except Exception as yfe:  # noqa: F841
	yf = None


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def fetch_prices_yfinance(ticker: str, period_days: int = 365 * 2) -> pd.DataFrame:
	if yf is None:
		raise RuntimeError("yfinance is required to fetch data automatically. Install yfinance or provide CSV.")
	end = datetime.utcnow().date()
	start = end - timedelta(days=period_days)
	df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True)
	if not isinstance(df, pd.DataFrame) or df.empty:
		raise RuntimeError(f"No data returned from yfinance for {ticker}")
	# Ensure daily frequency and forward fill
	df = df.asfreq("B").ffill()
	return df


def read_prices_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	# Try to infer date column
	date_col = None
	for cand in ["Date", "date", "Datetime", "datetime", "time", "Time"]:
		if cand in df.columns:
			date_col = cand
			break
	if date_col is None:
		raise ValueError("CSV must contain a date-like column (e.g., Date)")
	df[date_col] = pd.to_datetime(df[date_col])
	df = df.set_index(date_col).sort_index()
	# Expect a Close column
	close_col = None
	for cand in ["Close", "close", "Adj Close", "adj_close", "AdjClose", "Price", "price"]:
		if cand in df.columns:
			close_col = cand
			break
	if close_col is None:
		raise ValueError("CSV must contain a price column (e.g., Close)")
	# Business day frequency
	df = df.asfreq("B").ffill()
	# Keep only close
	return df[[close_col]].rename(columns={close_col: "Close"})


def load_close_series(df: pd.DataFrame) -> np.ndarray:
	if "Close" not in df.columns:
		raise ValueError("DataFrame must contain 'Close' column")
	series = df["Close"].astype(float).to_numpy()
	return series


def predict_with_chronos(context: np.ndarray, prediction_length: int = 14, device: str | None = None):
	# Lazy import to keep CLI startup fast if dependencies missing
	from chronos import BaseChronosPipeline

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	# Convert to torch tensor with expected shape [batch, context_length]
	context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

	pipeline = BaseChronosPipeline.from_pretrained(
		"amazon/chronos-bolt-base",
		device_map=device,
		torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32,
	)

	quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
	quantiles, mean = pipeline.predict_quantiles(
		context=context_tensor,
		prediction_length=prediction_length,
		quantile_levels=quantile_levels,
	)
	# quantiles: List[Tensor] or Tensor depending on version; normalize outputs
	if isinstance(mean, torch.Tensor):
		mean_np = mean.squeeze(0).detach().cpu().numpy()
	else:
		mean_np = np.asarray(mean)

	quantiles_out = {}
	if isinstance(quantiles, (list, tuple)):
		for q_level, q_tensor in zip(quantile_levels, quantiles):
			q_np = q_tensor.squeeze(0).detach().cpu().numpy() if isinstance(q_tensor, torch.Tensor) else np.asarray(q_tensor)
			quantiles_out[str(q_level)] = q_np.tolist()
	elif isinstance(quantiles, torch.Tensor):
		# assume shape [num_q, batch, pred_len] or [batch, num_q, pred_len]
		arr = quantiles.detach().cpu().numpy()
		if arr.ndim == 3 and arr.shape[1] == 1:
			arr = arr[:, 0, :]
		if arr.shape[0] == len(quantile_levels):
			for i, q_level in enumerate(quantile_levels):
				quantiles_out[str(q_level)] = arr[i].tolist()
		else:
			for i, q_level in enumerate(quantile_levels[: arr.shape[0]]):
				quantiles_out[str(q_level)] = arr[i].tolist()
	else:
		# Fallback
		for i, q_level in enumerate(quantile_levels):
			quantiles_out[str(q_level)] = []

	return mean_np.tolist(), quantiles_out


def plot_forecast(dates_hist: pd.DatetimeIndex, series_hist: np.ndarray, dates_fcst: pd.DatetimeIndex, mean_fcst: np.ndarray, out_path: str, title: str) -> None:
	plt.figure(figsize=(10, 5))
	plt.plot(dates_hist, series_hist, label="History", color="#1f77b4")
	plt.plot(dates_fcst, mean_fcst, label="Forecast (mean)", color="#d62728")
	plt.title(title)
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def next_business_days(start_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
	# Generate next n business days starting after start_date
	rng = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=n)
	return rng


def run_for_ticker(ticker: str, output_dir: str, source: str, csv_path: str | None, context_days: int, prediction_length: int, device: str | None) -> dict:
	if source == "yfinance":
		df = fetch_prices_yfinance(ticker)
	else:
		if not csv_path:
			raise ValueError("csv_path is required when source=csv")
		df = read_prices_csv(csv_path)

	# Close series
	series = load_close_series(df)
	# Keep only the last context_days for speed/stability
	if context_days > 0 and len(series) > context_days:
		series = series[-context_days:]

	mean_pred, quantiles = predict_with_chronos(series, prediction_length=prediction_length, device=device)

	# Build output
	ensure_dir(output_dir)
	last_date = df.index[-1]
	fcst_dates = next_business_days(last_date, prediction_length)

	# Save CSV
	csv_file = os.path.join(output_dir, f"{ticker}_forecast.csv")
	csv_df = pd.DataFrame({
		"date": fcst_dates,
		"forecast_mean": mean_pred,
	})
	for q_key, q_vals in quantiles.items():
		csv_df[f"q_{q_key}"] = q_vals if len(q_vals) == prediction_length else [np.nan] * prediction_length
	csv_df.to_csv(csv_file, index=False)

	# Save plot
	png_file = os.path.join(output_dir, f"{ticker}_forecast.png")
	plot_forecast(df.index[-len(series):], series[-len(series):], fcst_dates, np.array(mean_pred), png_file, f"{ticker} 14-day Forecast (Chronos-Bolt Base)")

	# Build JSON report
	report = {
		"ticker": ticker,
		"model": "amazon/chronos-bolt-base",
		"generated_at_utc": datetime.utcnow().isoformat() + "Z",
		"prediction_length": prediction_length,
		"context_length": int(len(series)),
		"csv_path": csv_file,
		"plot_path": png_file,
		"mean": mean_pred,
		"quantiles": quantiles,
	}
	json_file = os.path.join(output_dir, f"{ticker}_forecast_report.json")
	with open(json_file, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Chronos-Bolt Base 14-day forecast for NVDA and AAPL")
	parser.add_argument("--tickers", nargs="*", default=["NVDA", "AAPL"], help="Tickers to forecast")
	parser.add_argument("--prediction_length", type=int, default=14, help="Forecast horizon in days")
	parser.add_argument("--context_days", type=int, default=730, help="How many past business days to use as context")
	parser.add_argument("--output_dir", type=str, default=os.path.join("demo_reports", "chronos_bolt_base"), help="Directory to save reports")
	parser.add_argument("--source", choices=["yfinance", "csv"], default="yfinance", help="Data source")
	parser.add_argument("--csv_dir", type=str, default=None, help="Directory with CSV files named <TICKER>.csv when source=csv")
	parser.add_argument("--device", type=str, default=None, help="Force device 'cpu' or 'cuda' (auto if not set)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	ensure_dir(args.output_dir)
	all_reports = {}
	for ticker in args.tickers:
		csv_path = None
		if args.source == "csv":
			if not args.csv_dir:
				raise ValueError("--csv_dir is required when --source csv")
			csv_path = os.path.join(args.csv_dir, f"{ticker}.csv")
		report = run_for_ticker(
			ticker=ticker,
			output_dir=args.output_dir,
			source=args.source,
			csv_path=csv_path,
			context_days=args.context_days,
			prediction_length=args.prediction_length,
			device=args.device,
		)
		all_reports[ticker] = report

	# Save combined report
	combined_path = os.path.join(args.output_dir, "combined_reports.json")
	with open(combined_path, "w", encoding="utf-8") as f:
		json.dump(all_reports, f, indent=2)
	print(f"Saved combined report to {combined_path}")


if __name__ == "__main__":
	main()
