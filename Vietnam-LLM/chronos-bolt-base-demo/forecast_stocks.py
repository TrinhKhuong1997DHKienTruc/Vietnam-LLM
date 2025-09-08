import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

try:
	from chronos import ChronosPipeline
except ImportError as exc:
	raise SystemExit("chronos-forecasting is not installed. Please run: pip install chronos-forecasting") from exc

try:
	import yfinance as yf
except ImportError as exc:
	raise SystemExit("yfinance is not installed. Please run: pip install yfinance") from exc


def fetch_history(ticker: str, start: str, end: str) -> pd.Series:
	data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
	if data.empty:
		raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")
	return data["Close"].astype(float)


def ensure_output_dirs(base_dir: str, tickers: List[str]) -> Dict[str, str]:
	paths: Dict[str, str] = {}
	os.makedirs(base_dir, exist_ok=True)
	for t in tickers:
		p = os.path.join(base_dir, t)
		os.makedirs(p, exist_ok=True)
		paths[t] = p
	return paths


def load_pipeline(device_preference: str = "auto") -> ChronosPipeline:
	device_map = "cuda" if (device_preference == "auto" and torch.cuda.is_available()) else device_preference
	if device_map == "auto":
		device_map = "cpu"
	# Prefer bfloat16 on GPU, float32 on CPU
	torch_dtype = torch.bfloat16 if (device_map == "cuda") else torch.float32
	pipeline = ChronosPipeline.from_pretrained(
		"amazon/chronos-bolt-base",
		device_map=device_map,
		torch_dtype=torch_dtype,
	)
	return pipeline


def make_forecast(
	pipeline: ChronosPipeline,
	series: pd.Series,
	prediction_length: int,
	num_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	context = series.to_numpy(dtype=np.float32)
	# Model expects 1D context
	forecast_samples = pipeline.predict(
		context=context,
		prediction_length=prediction_length,
		num_samples=num_samples,
	)
	# forecast_samples shape: (num_samples, prediction_length)
	p50 = np.median(forecast_samples, axis=0)
	p10 = np.percentile(forecast_samples, 10, axis=0)
	p90 = np.percentile(forecast_samples, 90, axis=0)
	return p10, p50, p90


def save_csv(out_dir: str, ticker: str, dates: List[pd.Timestamp], p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> str:
	df = pd.DataFrame({
		"date": pd.to_datetime(dates),
		"p10": p10,
		"p50": p50,
		"p90": p90,
	})
	csv_path = os.path.join(out_dir, f"{ticker}_14day_forecast.csv")
	df.to_csv(csv_path, index=False)
	return csv_path


def save_json(out_dir: str, ticker: str, meta: Dict) -> str:
	json_path = os.path.join(out_dir, f"{ticker}_14day_report.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	return json_path


def save_plot(out_dir: str, ticker: str, history: pd.Series, dates: List[pd.Timestamp], p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> str:
	plt.figure(figsize=(11, 6))
	plt.plot(history.index, history.values, label=f"{ticker} Close", color="#1f77b4")
	plt.plot(dates, p50, label="Forecast p50", color="#ff7f0e")
	plt.fill_between(dates, p10, p90, color="#ff7f0e", alpha=0.2, label="Forecast 10-90%")
	plt.title(f"{ticker} 14-Day Forecast (Chronos-Bolt Base)")
	plt.xlabel("Date")
	plt.ylabel("Price (USD)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	png_path = os.path.join(out_dir, f"{ticker}_14day_forecast.png")
	plt.savefig(png_path, dpi=160)
	plt.close()
	return png_path


def run(tickers: List[str], days_history: int, prediction_length: int, output_base: str, device: str) -> None:
	end_date_dt = datetime.utcnow().date()
	start_date_dt = end_date_dt - timedelta(days=days_history)
	start_date = start_date_dt.isoformat()
	end_date = (end_date_dt + timedelta(days=1)).isoformat()

	out_map = ensure_output_dirs(output_base, tickers)
	pipeline = load_pipeline(device_preference=device)

	combined_index = []
	combined_rows = []

	for t in tickers:
		series = fetch_history(t, start=start_date, end=end_date)
		p10, p50, p90 = make_forecast(pipeline, series, prediction_length=prediction_length)
		forecast_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=prediction_length, freq="D")

		csv_path = save_csv(out_map[t], t, list(forecast_dates), p10, p50, p90)
		png_path = save_plot(out_map[t], t, series, list(forecast_dates), p10, p50, p90)

		meta = {
			"ticker": t,
			"model": "amazon/chronos-bolt-base",
			"generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
			"history_start": start_date,
			"history_end": str(series.index[-1].date()),
			"prediction_length": prediction_length,
			"artifacts": {
				"csv": csv_path,
				"plot": png_path,
			},
		}
		json_path = save_json(out_map[t], t, meta)

		for i in range(prediction_length):
			combined_index.append((t, forecast_dates[i]))
			combined_rows.append({"ticker": t, "date": forecast_dates[i], "p10": float(p10[i]), "p50": float(p50[i]), "p90": float(p90[i])})

	# Save combined CSV at base
	combined_df = pd.DataFrame(combined_rows)
	combined_csv = os.path.join(output_base, "combined_14day_forecasts.csv")
	combined_df.to_csv(combined_csv, index=False)

	# Top-level run metadata
	top_meta = {
		"tickers": tickers,
		"model": "amazon/chronos-bolt-base",
		"generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
		"history_days": days_history,
		"prediction_length": prediction_length,
		"combined_csv": combined_csv,
	}
	with open(os.path.join(output_base, "run_report.json"), "w", encoding="utf-8") as f:
		json.dump(top_meta, f, ensure_ascii=False, indent=2)

	print(f"Saved combined CSV: {combined_csv}")
	print(f"Saved per-ticker reports under: {output_base}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Forecast NVDA & AAPL 14 days using amazon/chronos-bolt-base")
	parser.add_argument("--tickers", type=str, default="NVDA,AAPL", help="Comma-separated tickers, default: NVDA,AAPL")
	parser.add_argument("--history_days", type=int, default=365*3, help="Number of past days to fetch for context")
	parser.add_argument("--prediction_length", type=int, default=14, help="Forecast horizon in days")
	parser.add_argument("--output", type=str, default=os.path.join("demo_reports", "chronos_bolt_base"), help="Output base directory for reports")
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device preference")
	args = parser.parse_args()

	tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
	run(tickers, args.history_days, args.prediction_length, args.output, args.device)
