from __future__ import annotations
import argparse
import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
from rich import print

from .config import paths
from .data_sources import get_daily_prices
from .forecasting import forecast_next_14_days
from .llm_reports import generate_report


def ensure_symbol_dir(symbol: str) -> str:
	dir_path = os.path.join(paths.root_reports_dir, symbol.upper())
	os.makedirs(dir_path, exist_ok=True)
	return dir_path


def save_csv(df: pd.DataFrame, out_path: str) -> None:
	df.to_csv(out_path, index=False)


def plot_forecast(symbol: str, hist: pd.DataFrame, future_dates, preds, out_path: str) -> None:
	plt.figure(figsize=(10, 5))
	plt.plot(hist["date"], hist["close"], label="Historical Close")
	plt.plot(future_dates, preds, label="Forecast Close", linestyle="--")
	plt.title(f"{symbol} - 14 Day Forecast")
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def run_for_symbol(symbol: str, history_days: int = 365):
	print(f"[bold green]Processing[/] {symbol} ...")
	dir_path = ensure_symbol_dir(symbol)
	# Fetch data
	hist = get_daily_prices(symbol, days=history_days)
	save_csv(hist, os.path.join(dir_path, f"{symbol}_historical.csv"))
	# Forecast
	hist = hist.sort_values("date")
	hist.set_index("date", inplace=True)
	fc = forecast_next_14_days(hist["close"])
	forecast_df = pd.DataFrame({
		"date": fc.future_dates,
		"predicted_close": fc.predicted_close,
	})
	save_csv(forecast_df, os.path.join(dir_path, f"{symbol}_forecast.csv"))
	plot_forecast(symbol, hist.reset_index(), fc.future_dates, fc.predicted_close, os.path.join(dir_path, f"{symbol}_forecast_plot.png"))
	# Report
	stats = {
		"history_days": history_days,
		"last_close": round(fc.last_close, 4),
		"mean_close_30d": round(float(hist["close"].tail(30).mean()), 4),
		"volatility_30d": round(float(hist["close"].tail(30).pct_change().std()), 6),
	}
	report_text = generate_report(symbol, stats, {
		"dates": [d.strftime("%Y-%m-%d") for d in fc.future_dates],
		"predicted_close": fc.predicted_close,
		"last_close": fc.last_close,
	})
	with open(os.path.join(dir_path, f"{symbol}_report.txt"), "w", encoding="utf-8") as f:
		f.write(report_text)
	print(f"[bold blue]Done[/] {symbol} -> {dir_path}")


def main():
	parser = argparse.ArgumentParser(description="StockAgent RT demo runner")
	parser.add_argument("--symbols", nargs="*", default=["NVDA", "AAPL"], help="Symbols to process")
	parser.add_argument("--history-days", type=int, default=365, help="Lookback days")
	args = parser.parse_args()
	for sym in args.symbols:
		run_for_symbol(sym.upper(), history_days=args.history_days)

if __name__ == "__main__":
	main()
