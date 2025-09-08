import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def ensure_output_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def load_model(model_dir_or_id: str = "TheFinAI/Fin-o1-8B") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
	local_dir = os.path.join(os.path.dirname(__file__), "Fin-o1-8B")
	if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
		model_source = local_dir
	else:
		model_source = model_dir_or_id

	tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=os.path.isdir(local_dir))
	model = AutoModelForCausalLM.from_pretrained(
		model_source,
		torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
		low_cpu_mem_usage=True,
		device_map="auto" if torch.cuda.is_available() else None,
	)
	if torch.cuda.is_available():
		model.to("cuda")
	return tokenizer, model


def trading_days_ahead(num_days: int) -> List[datetime]:
	# Use pandas_market_calendars if available; fallback to business days
	try:
		import pandas_market_calendars as mcal
		nyse = mcal.get_calendar("NYSE")
		start = datetime.utcnow().date()
		end = start + timedelta(days=30)
		schedule = nyse.schedule(start_date=start, end_date=end)
		dates = [ts.to_pydatetime().date() for ts in schedule.index]
		return dates[:num_days]
	except Exception:
		dates = []
		current = datetime.utcnow().date()
		while len(dates) < num_days:
			current += timedelta(days=1)
			if current.weekday() < 5:
				dates.append(current)
		return dates


def fetch_history(symbol: str, period: str = "180d") -> pd.DataFrame:
	data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
	if data.empty:
		raise RuntimeError(f"No data returned for {symbol}")
	return data


def build_prompt(symbol: str, history: pd.DataFrame, horizon: int) -> str:
	recent = history.tail(60).reset_index()
	recent_rows = []
	for _, row in recent.iterrows():
		date_str = row["Date"].strftime("%Y-%m-%d") if not isinstance(row["Date"], str) else row["Date"]
		recent_rows.append(
			{
				"date": date_str,
				"open": round(float(row["Open"]), 4),
				"high": round(float(row["High"]), 4),
				"low": round(float(row["Low"]), 4),
				"close": round(float(row["Close"]), 4),
				"volume": int(row["Volume"]),
			}
		)
	context = json.dumps({"symbol": symbol, "recent_ohlcv": recent_rows}, ensure_ascii=False)
	future_dates = [d.strftime("%Y-%m-%d") for d in trading_days_ahead(horizon)]
	instruction = (
		"You are Fin-o1-8B, a financial reasoning LLM. "
		"Given recent OHLCV data, produce a conservative 14-trading-day forecast of daily closing prices. "
		"Also provide a concise rationale and risk factors. Return a single strict JSON object with keys: "
		"'forecast' (array of {date, close}), 'analysis' (string), 'assumptions' (string). "
		f"Use these exact future dates: {future_dates}."
	)
	prompt = f"<context>{context}</context>\n<task>{instruction}</task>\n<format>JSON only</format>"
	return prompt


def generate_json(tokenizer, model, prompt: str, max_new_tokens: int = 1024) -> Dict:
	inputs = tokenizer(prompt, return_tensors="pt")
	if torch.cuda.is_available():
		inputs = {k: v.to("cuda") for k, v in inputs.items()}
	with torch.no_grad():
		output_ids = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			temperature=0.2,
			top_p=0.9,
			do_sample=True,
		)
	text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	# Try to extract JSON
	match = re.search(r"\{[\s\S]*\}", text)
	candidate = match.group(0) if match else text
	# Remove potential trailing commentary
	candidate = candidate.strip()
	try:
		return json.loads(candidate)
	except Exception:
		# Fallback: attempt to extract code block
		code_match = re.search(r"```(json)?([\s\S]*?)```", text)
		if code_match:
			code = code_match.group(2).strip()
			return json.loads(code)
		raise ValueError(f"Model output not valid JSON.\n---\n{text}\n---")


def save_outputs(symbol: str, result: Dict, out_dir: str) -> Tuple[str, str, str]:
	ensure_output_dir(out_dir)
	# JSON report
	json_path = os.path.join(out_dir, f"{symbol.lower()}_14day_report.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)
	# CSV forecast
	forecast = result.get("forecast", [])
	csv_path = os.path.join(out_dir, f"{symbol.lower()}_14day_forecast.csv")
	pd.DataFrame(forecast).to_csv(csv_path, index=False)
	# Plot
	df = pd.DataFrame(forecast)
	plt.figure(figsize=(10, 5))
	plt.plot(pd.to_datetime(df["date"]), df["close"], marker="o", label=f"{symbol} forecast")
	plt.title(f"{symbol} 14-day Forecast (Fin-o1-8B)")
	plt.xlabel("Date")
	plt.ylabel("Close (forecast)")
	plt.grid(True, alpha=0.3)
	plt.xticks(rotation=45)
	plt.tight_layout()
	png_path = os.path.join(out_dir, f"{symbol.lower()}_14day_forecast.png")
	plt.savefig(png_path)
	plt.close()
	return json_path, csv_path, png_path


def run_for_symbol(tokenizer, model, symbol: str, horizon: int, out_root: str) -> Dict[str, str]:
	print(f"Processing {symbol}...")
	history = fetch_history(symbol)
	prompt = build_prompt(symbol, history, horizon)
	result = generate_json(tokenizer, model, prompt)
	out_dir = os.path.join(out_root, symbol.lower())
	json_path, csv_path, png_path = save_outputs(symbol, result, out_dir)
	# Save a short analysis text
	analysis_path = os.path.join(out_dir, f"{symbol.lower()}_analysis.txt")
	with open(analysis_path, "w", encoding="utf-8") as f:
		f.write(result.get("analysis", ""))
		f.write("\n\nAssumptions:\n")
		f.write(result.get("assumptions", ""))
	return {
		"json": json_path,
		"csv": csv_path,
		"png": png_path,
		"analysis": analysis_path,
	}


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Fin-o1-8B 14-day stock forecast and report generator")
	parser.add_argument("--symbols", nargs="*", default=["NVDA", "AAPL"], help="Ticker symbols")
	parser.add_argument("--horizon", type=int, default=14, help="Number of trading days to forecast")
	parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "demo"), help="Output directory root")
	parser.add_argument("--model", type=str, default="TheFinAI/Fin-o1-8B", help="Model id or local dir")
	args = parser.parse_args()

	ensure_output_dir(args.out)
	print("Loading model...")
	tokenizer, model = load_model(args.model)

	artifacts = {}
	for sym in tqdm(args.symbols):
		artifacts[sym] = run_for_symbol(tokenizer, model, sym, args.horizon, args.out)

	# Save a combined metadata file
	meta = {
		"generated_at": datetime.utcnow().isoformat() + "Z",
		"symbols": args.symbols,
		"horizon": args.horizon,
		"artifacts": artifacts,
	}
	with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print("Done. Outputs written to:", args.out)


if __name__ == "__main__":
	main()
