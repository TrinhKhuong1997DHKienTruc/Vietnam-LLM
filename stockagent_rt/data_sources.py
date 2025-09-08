from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import market_keys

USER_AGENT = {
	"User-Agent": "StockAgent-RT/1.0 (+https://example.com)"
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get(url: str, params: dict) -> requests.Response:
	resp = requests.get(url, params=params, headers=USER_AGENT, timeout=30)
	resp.raise_for_status()
	return resp

def fetch_finnhub_daily(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
	if not market_keys.finnhub_api_key:
		return None
	end = int(time.time())
	start = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp())
	url = "https://finnhub.io/api/v1/stock/candle"
	params = {"symbol": symbol, "resolution": "D", "from": start, "to": end, "token": market_keys.finnhub_api_key}
	try:
		r = _get(url, params)
		data = r.json()
		if data.get("s") != "ok":
			return None
		df = pd.DataFrame({
			"date": pd.to_datetime(pd.Series(data["t"]), unit="s", utc=True).dt.tz_convert(None),
			"open": data["o"],
			"high": data["h"],
			"low": data["l"],
			"close": data["c"],
			"volume": data["v"],
		})
		df = df.sort_values("date").reset_index(drop=True)
		return df.tail(days)
	except Exception:
		return None

def fetch_fmp_daily(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
	if not market_keys.fmp_api_key:
		return None
	url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
	params = {"serietype": "line", "timeseries": days * 2, "apikey": market_keys.fmp_api_key}
	try:
		r = _get(url, params)
		js = r.json()
		hist = js.get("historical") or js.get("historicalStockList") or []
		if not hist:
			return None
		df = pd.DataFrame(hist)
		if "date" not in df or "close" not in df:
			return None
		df["date"] = pd.to_datetime(df["date"])
		df = df.sort_values("date").reset_index(drop=True)
		# Normalize columns
		if "open" not in df:
			df["open"] = df["close"]
		if "high" not in df:
			df["high"] = df["close"]
		if "low" not in df:
			df["low"] = df["close"]
		if "volume" not in df:
			df["volume"] = 0
		return df.tail(days)
	except Exception:
		return None

def get_daily_prices(symbol: str, days: int = 365) -> pd.DataFrame:
	df = fetch_finnhub_daily(symbol, days)
	if df is None:
		df = fetch_fmp_daily(symbol, days)
	if df is None or df.empty:
		raise RuntimeError(f"Failed to fetch data for {symbol}")
	cols = ["date", "open", "high", "low", "close", "volume"]
	return df[cols]
