#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import requests
import yfinance as yf

from utils import generate_combined_state, stock_close_prices
from agents.DQN import Agent as DQNAgent


def fetch_fmp_daily(symbol: str, apikey: Optional[str]) -> Optional[pd.DataFrame]:
	if not apikey:
		return None
	# Fetch last 6 months daily OHLC
	to_date = datetime.utcnow().date()
	from_date = to_date - timedelta(days=200)
	url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={from_date}&to={to_date}&apikey={apikey}"
	try:
		resp = requests.get(url, timeout=20)
		resp.raise_for_status()
		data = resp.json()
		hist = data.get('historical') or []
		if not hist:
			return None
		df = pd.DataFrame(hist)
		# Ensure expected columns
		rename_map = {
			'date': 'Date',
			'open': 'Open',
			'high': 'High',
			'low': 'Low',
			'close': 'Close',
			'adjClose': 'Adj Close',
			'volume': 'Volume'
		}
		df.rename(columns=rename_map, inplace=True)
		df = df[['Date','Open','High','Low','Close','Adj Close','Volume']]
		df.sort_values('Date', inplace=True)
		return df
	except Exception:
		return None


def ensure_data_csv(ticker_key: str, fmp_key: Optional[str]) -> str:
	"""Ensure a CSV exists under data/{ticker_key}.csv; if missing, fetch via yfinance then FMP."""
	csv_path = os.path.join('data', f'{ticker_key}.csv')
	if os.path.exists(csv_path):
		return csv_path
	# Attempt to fetch by stripping suffixes like _2018
	symbol = ticker_key.split('_')[0]
	# Try yfinance first
	df = yf.download(symbol, period='6mo', interval='1d', auto_adjust=False, progress=False)
	if df.empty:
		# Try FMP fallback
		df = fetch_fmp_daily(symbol, fmp_key)
		if df is None or (hasattr(df, 'empty') and df.empty):
			raise RuntimeError(f'No data fetched for {symbol}')
	else:
		df = df.reset_index()
		df.rename(columns={
			'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low',
			'Close': 'Close', 'Adj Close': 'Adj Close' if 'Adj Close' in df.columns else 'Close', 'Volume': 'Volume'
		}, inplace=True)
		# Ensure required columns
		for col in ['Date','Open','High','Low','Close','Adj Close','Volume']:
			if col not in df.columns:
				if col == 'Adj Close':
					df[col] = df['Close']
				else:
					df[col] = np.nan
	# Save in expected format
	os.makedirs('data', exist_ok=True)
	df[['Date','Open','High','Low','Close','Adj Close','Volume']].to_csv(csv_path, index=False)
	return csv_path


def simple_next_week_forecast(prices, window_size, agent):
	# Use the last window to simulate the agent's preferred action distribution
	end_index = len(prices) - 1
	state = generate_combined_state(end_index, window_size, prices, agent.balance, len(agent.inventory))
	raw = agent.model.predict(agent._prepare_state(state))[0]
	probs = np.array(raw).reshape(-1)
	if probs.size < 3:
		# Fallback: approximate as neutral probabilities
		probs = np.array([0.34, 0.33, 0.33])
	# naive forecast: extrapolate last close by weighted signal
	last_close = prices[-1]
	buy_signal = probs[1]
	sell_signal = probs[2]
	net = float(buy_signal - sell_signal)
	# scale change by recent volatility
	recent = np.array(prices[-min(20, len(prices)) :])
	vol = float(np.std(np.diff(recent)) if len(recent) > 1 else 0.0)
	delta = net * (vol if vol > 0 else last_close * 0.002)
	forecast = [float(last_close + (i + 1) * delta) for i in range(5)]
	return probs.tolist(), forecast


def maybe_llm_summary(base_url: Optional[str], api_key: Optional[str], model: Optional[str], payload: Dict[str, Any]) -> Optional[str]:
	if not base_url or not api_key or not model:
		return None
	try:
		url = base_url.rstrip('/') + '/v1/chat/completions'
		headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
		messages = [
			{"role":"system","content":"You are a financial assistant. Create a concise, neutral 1-paragraph summary of the next-week forecast."},
			{"role":"user","content": json.dumps(payload)}
		]
		resp = requests.post(url, headers=headers, json={"model": model, "messages": messages, "temperature": 0.2}, timeout=30)
		resp.raise_for_status()
		data = resp.json()
		# OpenAI-style parsing
		return data.get('choices',[{}])[0].get('message',{}).get('content')
	except Exception:
		return None


def generate_report(ticker_key, model_path, window_size, initial_balance, out_dir, llm_cfg):
	os.makedirs(out_dir, exist_ok=True)
	csv = ensure_data_csv(ticker_key, llm_cfg.get('FMP_API_KEY'))
	prices = stock_close_prices(ticker_key)
	agent = DQNAgent(state_dim=window_size + 3, balance=initial_balance, is_eval=True, model_name=os.path.basename(model_path).replace('.h5',''))
	probs, forecast = simple_next_week_forecast(prices, window_size, agent)
	last_close = prices[-1]
	start_date = datetime.utcnow().date()
	dates = [(start_date + timedelta(days=i+1)).isoformat() for i in range(5)]
	payload = {
		"ticker": ticker_key,
		"last_close": float(last_close),
		"action_probabilities": {"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])},
		"forecast_horizon_days": 5,
		"forecast": [{"date": d, "price": float(p)} for d, p in zip(dates, forecast)],
	}
	summary = maybe_llm_summary(llm_cfg.get('base_url'), llm_cfg.get('api_key'), llm_cfg.get('model'), payload)
	if summary:
		payload['summary'] = summary
	out_path = os.path.join(out_dir, f"forecast_{ticker_key}_{start_date.isoformat()}.json")
	with open(out_path, 'w') as f:
		json.dump(payload, f, indent=2)
	print(f"Saved {out_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--stocks", nargs='+', required=True, help="Tickers or CSV keys inside data/, e.g., NVDA MSFT NVDA_2018")
	parser.add_argument("--model_to_load", default="DQN_ep10.h5", help="Model file under saved_models/")
	parser.add_argument("--window_size", type=int, default=10)
	parser.add_argument("--initial_balance", type=int, default=50000)
	parser.add_argument("--out_dir", default="reports")
	args = parser.parse_args()

	load_dotenv()
	llm_cfg = {
		"model": os.getenv('LLM_MODEL'),
		"api_key": os.getenv('LLM_API_KEY'),
		"base_url": os.getenv('LLM_BASE_URL'),
		"FMP_API_KEY": os.getenv('FMP_API_KEY')
	}

	model_path = os.path.join('saved_models', args.model_to_load)
	for s in args.stocks:
		generate_report(s, model_path, args.window_size, args.initial_balance, args.out_dir, llm_cfg)