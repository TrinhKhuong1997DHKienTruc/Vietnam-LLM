from __future__ import annotations
from typing import Dict, Any

from .config import llm_config

REPORT_PROMPT = (
	"You are a financial analyst. Given a symbol, recent metrics, and a 14-day forecast, "
	"write a concise report with: (1) market context, (2) key recent performance, "
	"(3) forecast summary with risks, (4) simple recommendation (Not investment advice)."
)


def _use_openai(messages: list[dict[str, str]]) -> str:
	from openai import OpenAI
	base_url = llm_config.openai_base_url
	client = OpenAI(api_key=llm_config.openai_api_key, base_url=base_url) if base_url else OpenAI(api_key=llm_config.openai_api_key)
	resp = client.chat.completions.create(
		model=llm_config.openai_model,
		messages=messages,
		temperature=0.4,
		max_tokens=800,
	)
	return resp.choices[0].message.content


def _use_gemini(prompt: str) -> str:
	import google.generativeai as genai
	genai.configure(api_key=llm_config.google_api_key)
	model = genai.GenerativeModel(llm_config.gemini_model)
	resp = model.generate_content(prompt)
	return resp.text


def generate_report(symbol: str, stats: Dict[str, Any], forecast: Dict[str, Any]) -> str:
	bullet_stats = "\n".join([f"- {k}: {v}" for k, v in stats.items()])
	f_dates = ", ".join(forecast.get("dates", [])[:5]) + (" ..." if len(forecast.get("dates", [])) > 5 else "")
	prompt = (
		f"{REPORT_PROMPT}\n\n"
		f"Symbol: {symbol}\n"
		f"Recent Stats:\n{bullet_stats}\n\n"
		f"Forecast: next 14 business days. First dates: {f_dates}. "
		f"Last close: {forecast.get('last_close')}\n"
	)
	if llm_config.openai_api_key:
		messages = [
			{"role": "system", "content": "You are a precise financial analyst."},
			{"role": "user", "content": prompt},
		]
		return _use_openai(messages)
	elif llm_config.google_api_key:
		return _use_gemini(prompt)
	else:
		return (
			"[LLM disabled] Provide API keys to generate a narrative.\n"
			f"Symbol {symbol}: last_close={forecast.get('last_close')}, "
			f"next_days={len(forecast.get('dates', []))}."
		)
