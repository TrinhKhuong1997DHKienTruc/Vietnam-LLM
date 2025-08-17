#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import time
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


FMP_PRICE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&apikey={api_key}"


@dataclass
class ForecastConfig:
	symbols: List[str]
	lookback_days: int = 365 * 3
	features_lags: int = 10
	target_horizon_days: int = 5  # approx next week trading days
	out_dir: str = "reports"


def fetch_daily_close_fmp(symbol: str, api_key: str, lookback_days: int) -> pd.DataFrame:
	url = FMP_PRICE_URL.format(symbol=symbol, api_key=api_key)
	r = requests.get(url, timeout=60)
	r.raise_for_status()
	data = r.json()
	if not isinstance(data, dict) or "historical" not in data:
		raise RuntimeError(f"Unexpected FMP response for {symbol}")
	df = pd.DataFrame(data["historical"])  # contains date, close, etc.
	df = df[["date", "close"]].dropna()
	df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
	df = df.sort_values("date").tail(lookback_days)
	df = df.reset_index(drop=True)
	return df


def make_features(df: pd.DataFrame, lags: int) -> pd.DataFrame:
	df = df.copy()
	for i in range(1, lags + 1):
		df[f"lag_{i}"] = df["close"].shift(i)
	# simple technicals
	df["ret_1"] = df["close"].pct_change(1)
	df["ret_5"] = df["close"].pct_change(5)
	df["ma_5"] = df["close"].rolling(5).mean()
	df["ma_10"] = df["close"].rolling(10).mean()
	df["std_5"] = df["close"].rolling(5).std()
	df["std_10"] = df["close"].rolling(10).std()
	# drop early NaNs
	df = df.dropna().reset_index(drop=True)
	return df


def build_feature_row_from_closes(feature_cols: List[str], last_closes: List[float], lags: int) -> np.ndarray:
	row = {}
	# lag_i features
	for i in range(1, lags + 1):
		row[f"lag_{i}"] = last_closes[-i] if len(last_closes) >= i else np.nan
	# technicals based on last known closes (include the most recent close)
	if len(last_closes) >= 2:
		row["ret_1"] = (last_closes[-1] / last_closes[-2]) - 1.0
	else:
		row["ret_1"] = 0.0
	if len(last_closes) >= 6:
		row["ret_5"] = (last_closes[-1] / last_closes[-6]) - 1.0
	else:
		row["ret_5"] = 0.0
	if len(last_closes) >= 5:
		arr5 = np.array(last_closes[-5:], dtype=float)
		row["ma_5"] = float(np.mean(arr5))
		row["std_5"] = float(np.std(arr5, ddof=0))
	else:
		row["ma_5"] = float(last_closes[-1])
		row["std_5"] = 0.0
	if len(last_closes) >= 10:
		arr10 = np.array(last_closes[-10:], dtype=float)
		row["ma_10"] = float(np.mean(arr10))
		row["std_10"] = float(np.std(arr10, ddof=0))
	else:
		row["ma_10"] = float(np.mean(last_closes))
		row["std_10"] = float(np.std(np.array(last_closes, dtype=float), ddof=0)) if len(last_closes) > 1 else 0.0
	# Build vector in the exact order of feature_cols
	return np.array([[row.get(c, 0.0) for c in feature_cols]], dtype=float)


def train_and_forecast_next_week(df_close: pd.DataFrame, lags: int, horizon_days: int):
	feat_df = make_features(df_close, lags)
	# Target: next-day close return; then compound to 5 days
	feat_df["target_ret_1"] = feat_df["close"].pct_change().shift(-1)
	feat_df = feat_df.dropna().reset_index(drop=True)
	feature_cols = [c for c in feat_df.columns if c not in ("date", "close", "target_ret_1")]
	X = feat_df[feature_cols].values
	y = feat_df["target_ret_1"].values
	if len(feat_df) < 100:
		raise RuntimeError("Not enough data after feature creation")
	# Simple split: last 60 samples as validation
	split = max(100, len(feat_df) - 60)
	X_train, y_train = X[:split], y[:split]
	X_val, y_val = X[split:], y[split:]
	# Models
	rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
	xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=0)
	rf.fit(X_train, y_train)
	xgb.fit(X_train, y_train)
	yrf = rf.predict(X_val)
	yxgb = xgb.predict(X_val)
	mae_rf = float(mean_absolute_error(y_val, yrf))
	mae_xgb = float(mean_absolute_error(y_val, yxgb))
	best = rf if mae_rf <= mae_xgb else xgb
	# Forecast next horizon_days via one-step recursion using last closes
	last_closes = feat_df["close"].tolist()
	preds = []
	est_closes = [last_closes[-1]]
	for _ in range(horizon_days):
		last_row = build_feature_row_from_closes(feature_cols, last_closes, lags)
		pred_ret1 = float(best.predict(last_row)[0])
		preds.append(pred_ret1)
		new_close = est_closes[-1] * (1.0 + pred_ret1)
		est_closes.append(new_close)
		last_closes.append(new_close)
	# Build forecast table
	start_date = feat_df["date"].iloc[-1] if "date" in feat_df.columns else df_close["date"].iloc[-1]
	forecast_dates = []
	curr_date = pd.to_datetime(start_date).normalize()
	added = 0
	while len(forecast_dates) < horizon_days and added < 21:
		curr_date += pd.Timedelta(days=1)
		if curr_date.weekday() < 5:
			forecast_dates.append(curr_date)
		added += 1
	est_closes = est_closes[1:1+horizon_days]
	fc_df = pd.DataFrame({"date": forecast_dates, "predicted_close": est_closes, "predicted_ret": preds[:len(forecast_dates)]})
	fc_df = fc_df.iloc[:horizon_days].reset_index(drop=True)
	metrics = {
		"val_mae_rf": mae_rf,
		"val_mae_xgb": mae_xgb,
		"best_model": "RandomForest" if best is rf else "XGBoost",
		"last_close": est_closes[0],
	}
	return fc_df, metrics


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def main():
	parser = argparse.ArgumentParser(description="Forecast next-week closes for given symbols using simple ML models (FMP data)")
	parser.add_argument("--symbols", type=str, default="MSFT,NVDA", help="Comma-separated tickers")
	parser.add_argument("--out_dir", type=str, default="reports", help="Output directory")
	parser.add_argument("--lookback_days", type=int, default=365*3)
	parser.add_argument("--lags", type=int, default=10)
	parser.add_argument("--horizon", type=int, default=5)
	args = parser.parse_args()

	load_dotenv()
	fmp_key = os.getenv("FMP_API_KEY")
	if not fmp_key:
		print("ERROR: FMP_API_KEY is required. Set it in environment or .env file.")
		sys.exit(1)

	symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
	cfg = ForecastConfig(symbols=symbols, lookback_days=args.lookback_days, features_lags=args.lags, target_horizon_days=args.horizon, out_dir=args.out_dir)

	ensure_dir(cfg.out_dir)
	all_reports = {}
	for sym in cfg.symbols:
		print(f"Fetching data for {sym}...")
		df = fetch_daily_close_fmp(sym, fmp_key, cfg.lookback_days)
		print(f"{sym}: {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")
		fc_df, metrics = train_and_forecast_next_week(df, cfg.features_lags, cfg.target_horizon_days)
		out_csv = os.path.join(cfg.out_dir, f"{sym}_forecast_next_week.csv")
		fc_df.to_csv(out_csv, index=False)
		print(f"Saved: {out_csv}")
		all_reports[sym] = {"metrics": metrics, "report_csv": out_csv}

	summary_path = os.path.join(cfg.out_dir, "summary.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(all_reports, f, indent=2, default=str)
	print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
	main()