from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import requests


@dataclass
class PreparedData:
	scaler: MinMaxScaler
	x_train: np.ndarray
	y_train: np.ndarray
	x_val: np.ndarray
	y_val: np.ndarray
	x_test: np.ndarray
	y_test: np.ndarray
	feature_columns: list[str]


def _fetch_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
	df = yf.download(ticker, start=start, end=end, progress=False)
	return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _fetch_fmp(ticker: str, start: str, end: str) -> pd.DataFrame:
	api_key = os.getenv("FMP_API_KEY")
	if not api_key:
		return pd.DataFrame()
	url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start}&to={end}&apikey={api_key}"
	r = requests.get(url, timeout=30)
	if r.status_code != 200:
		return pd.DataFrame()
	data = r.json()
	hist = data.get("historical") or []
	if not hist:
		return pd.DataFrame()
	df = pd.DataFrame(hist)
	# FMP returns newest-first
	df = df.sort_values("date")
	df["date"] = pd.to_datetime(df["date"])  # ensure datetime index
	df = df.rename(columns={
		"open": "open",
		"high": "high",
		"low": "low",
		"close": "close",
		"adjClose": "adj_close",
		"volume": "volume",
	})[["date", "open", "high", "low", "close", "adj_close", "volume"]]
	df = df.set_index("date")
	return df


def fetch_ohlcv_daily(ticker: str, lookback_years: int = 6) -> pd.DataFrame:
	end = datetime.utcnow().date()
	start = end - timedelta(days=365 * lookback_years + 30)
	start_s, end_s = start.isoformat(), end.isoformat()
	# Try yfinance
	df = _fetch_yf(ticker, start_s, end_s)
	if df is None or df.empty:
		# Try FMP fallback
		df = _fetch_fmp(ticker, start_s, end_s)
		if df.empty:
			raise RuntimeError(f"No data returned for ticker {ticker}")
		# already normalized by _fetch_fmp
		return df
	# Normalize yfinance result
	df = df.rename(columns={
		"Open": "open",
		"High": "high",
		"Low": "low",
		"Close": "close",
		"Adj Close": "adj_close",
		"Volume": "volume",
	})
	df.index.name = "date"
	return df


def train_val_test_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	n = len(df)
	train_end = int(n * train_ratio)
	val_end = int(n * (train_ratio + val_ratio))
	train = df.iloc[:train_end]
	val = df.iloc[train_end:val_end]
	test = df.iloc[val_end:]
	return train, val, test


def prepare_sequences(df: pd.DataFrame, lookback: int, target_col: str = "close") -> Tuple[np.ndarray, np.ndarray, list[str], MinMaxScaler]:
	feature_cols = ["open", "high", "low", "close", "volume"]
	features = df[feature_cols].astype(float).copy()
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(features.values)
	x, y = [], []
	for i in range(lookback, scaled.shape[0]):
		x.append(scaled[i - lookback:i, :])
		y.append(scaled[i, feature_cols.index(target_col)])
	return np.array(x), np.array(y), feature_cols, scaler


def build_datasets(df: pd.DataFrame, lookback: int = 60) -> PreparedData:
	train_df, val_df, test_df = train_val_test_split(df)
	x_train, y_train, feature_cols, scaler = prepare_sequences(train_df, lookback)
	scaled_val = scaler.transform(val_df[["open", "high", "low", "close", "volume"]].astype(float))
	scaled_test = scaler.transform(test_df[["open", "high", "low", "close", "volume"]].astype(float))
	# Build val sequences
	x_val, y_val = [], []
	for i in range(lookback, scaled_val.shape[0]):
		x_val.append(scaled_val[i - lookback:i, :])
		y_val.append(scaled_val[i, 3])
	# Build test sequences
	x_test, y_test = [], []
	for i in range(lookback, scaled_test.shape[0]):
		x_test.append(scaled_test[i - lookback:i, :])
		y_test.append(scaled_test[i, 3])
	return PreparedData(
		scaler=scaler,
		x_train=np.array(x_train),
		y_train=np.array(y_train),
		x_val=np.array(x_val),
		y_val=np.array(y_val),
		x_test=np.array(x_test),
		y_test=np.array(y_test),
		feature_columns=feature_cols,
	)