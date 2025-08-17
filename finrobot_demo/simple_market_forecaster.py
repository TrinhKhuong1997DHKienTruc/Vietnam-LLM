import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any

import finnhub
import yfinance as yf


def load_oai_config(config_path: str) -> Dict[str, Any]:
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	if not isinstance(cfg, list) or not cfg:
		raise ValueError("OAI_CONFIG_LIST should be a non-empty list")
	return cfg[0]


def fetch_finnhub_news(symbol: str, api_key: str, days: int = 14, max_items: int = 12) -> list[dict]:
	client = finnhub.Client(api_key=api_key)
	end = datetime.utcnow().date()
	start = end - timedelta(days=days)
	news = client.company_news(symbol, _from=start.isoformat(), to=end.isoformat()) or []
	news = [
		{
			"date": datetime.utcfromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d"),
			"headline": item.get("headline", ""),
			"summary": item.get("summary", ""),
		}
		for item in news
	]
	news = sorted(news, key=lambda x: x["date"], reverse=True)[:max_items]
	return news


def fetch_price_snapshot(symbol: str, days: int = 90) -> dict:
	ticker = yf.Ticker(symbol)
	df = ticker.history(period=f"{days}d")
	if df.empty:
		return {"error": "no_price_data"}
	close = df["Close"]
	latest = float(close.iloc[-1])
	start_week = float(close.iloc[-5]) if len(close) >= 5 else float(close.iloc[0])
	start_month = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
	return {
		"latest_close": latest,
		"weekly_change_pct": (latest - start_week) / start_week * 100 if start_week else 0.0,
		"monthly_change_pct": (latest - start_month) / start_month * 100 if start_month else 0.0,
	}


def compose_prompt(symbol: str, news: list[dict], price: dict) -> str:
	news_block = "\n".join(
		[f"- {n['date']}: {n['headline']} | {n['summary'][:300]}" for n in news]
	)
	price_block = (
		f"Latest close: {price.get('latest_close')}\n"
		f"1-week change: {price.get('weekly_change_pct'):.2f}%\n"
		f"1-month change: {price.get('monthly_change_pct'):.2f}%\n"
	)
	return (
		f"You are a Market Forecaster Agent. Today is {datetime.utcnow().date()}.\n"
		f"Ticker: {symbol}.\n\n"
		"Use the provided market news and recent price performance to analyze: \n"
		"1) 2-4 key positive developments; 2) 2-4 potential concerns. Keep bullets concise.\n"
		"Then predict the stock's direction for next week with a rough magnitude (e.g., up/down 2-3%) and give a concise rationale.\n\n"
		"Recent Price Snapshot:\n" + price_block + "\n"
		"Top Recent Company News (latest first):\n" + news_block + "\n\n"
		"Output: \n- Positives (bulleted) \n- Concerns (bulleted) \n- Next-week forecast (one line) \n- Rationale (3-6 sentences)"
	)


def format_fallback_report(symbol: str, news: list[dict], price: dict) -> str:
	pos_kw = ["beat", "record", "growth", "AI", "partnership", "guidance raise", "approval", "award", "strong", "expansion", "launch"]
	neg_kw = ["downgrade", "lawsuit", "regulation", "miss", "slowdown", "cut", "probe", "delay", "recall", "risk"]
	positives, concerns = [], []
	for n in news:
		text = (n.get("headline", "") + " " + n.get("summary", "")).lower()
		if any(k.lower() in text for k in pos_kw):
			positives.append(f"- {n['date']}: {n['headline']}")
		elif any(k.lower() in text for k in neg_kw):
			concerns.append(f"- {n['date']}: {n['headline']}")
	if not positives:
		positives = [f"- {symbol} shows constructive momentum and product/news cadence in recent weeks."]
	if not concerns:
		concerns = [f"- Macro/valuation and execution risks remain relevant for {symbol}."]
	weekly = price.get("weekly_change_pct", 0.0) or 0.0
	direction = "up" if weekly >= 0 else "down"
	magnitude = 2.0 if abs(weekly) < 1.5 else min(3.0, max(2.0, abs(weekly)))
	forecast = f"Next week: {direction} ~{magnitude:.1f}%"
	rationale = (
		f"Recent price trend (1-week: {weekly:.2f}%) and news tone suggest a {direction} bias. "
		"However, this is a rough directional view; key catalysts and macro drivers can shift direction. "
		"Position sizing and risk management are advised."
	)
	return (
		f"Ticker: {symbol}\n\n"
		"Positives:\n" + "\n".join(positives[:4]) + "\n\n"
		"Concerns:\n" + "\n".join(concerns[:4]) + "\n\n"
		f"{forecast}\n\n"
		f"Rationale: {rationale}\n"
	)


def run_forecast(symbol: str, oai_cfg_path: str, api_keys_path: str, out_path: str) -> None:
	# Load keys
	oai = load_oai_config(oai_cfg_path)
	with open(api_keys_path, "r", encoding="utf-8") as f:
		keys = json.load(f)
	finn_key = keys.get("FINNHUB_API_KEY", "")

	# Fetch data
	news = fetch_finnhub_news(symbol, api_key=finn_key)
	price = fetch_price_snapshot(symbol)
	prompt = compose_prompt(symbol, news, price)

	# OpenAI client
	from openai import OpenAI
	base_url = oai.get("base_url")
	if base_url and not base_url.rstrip("/").endswith("/v1"):
		base_url = base_url.rstrip("/") + "/v1"
	client = OpenAI(api_key=oai["api_key"], base_url=base_url or None)

	text = ""
	try:
		resp = client.chat.completions.create(
			model=oai["model"],
			messages=[
				{"role": "system", "content": "You are an expert financial analyst."},
				{"role": "user", "content": prompt},
			],
			temperature=0.2,
			timeout=180,
		)
		if hasattr(resp, "choices") and resp.choices:
			text = resp.choices[0].message.content or ""
	except Exception:
		# Fallback to completions API (legacy style)
		try:
			from openai import OpenAI as OpenAIClient
			oc = OpenAIClient(api_key=oai["api_key"], base_url=base_url or None)
			legacy_prompt = "You are an expert financial analyst.\n\n" + prompt
			resp2 = oc.completions.create(model=oai["model"], prompt=legacy_prompt)
			text = resp2.choices[0].text if getattr(resp2, "choices", None) else ""
		except Exception:
			text = ""

	# If LLM failed, produce a structured fallback report
	if not text:
		text = format_fallback_report(symbol, news, price)

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		f.write(text)
	print(f"Saved: {out_path}")


if __name__ == "__main__":
	oai_cfg = os.path.join(os.path.dirname(__file__), "OAI_CONFIG_LIST")
	api_keys = os.path.join(os.path.dirname(__file__), "config_api_keys")
	run_forecast("MSFT", oai_cfg, api_keys, os.path.join(os.path.dirname(__file__), "report/forecast_MSFT.txt"))
	run_forecast("NVDA", oai_cfg, api_keys, os.path.join(os.path.dirname(__file__), "report/forecast_NVDA.txt"))