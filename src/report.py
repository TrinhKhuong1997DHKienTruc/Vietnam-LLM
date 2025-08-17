from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>LSTM Forecast - {{ ticker }}</title>
	<style>
		body { font-family: Arial, sans-serif; margin: 24px; }
		img { max-width: 100%; height: auto; }
		table { border-collapse: collapse; }
		td, th { padding: 8px 12px; border: 1px solid #ddd; }
	</style>
</head>
<body>
	<h1>Forecast for {{ ticker }} (next 5 trading days)</h1>
	<p>Generated: {{ generated_at }}</p>
	<h2>Summary</h2>
	<table>
		<tr><th>Last Close</th><td>{{ last_close }}</td></tr>
		<tr><th>Forecast (scaled back)</th><td>{{ forecast_values }}</td></tr>
	</table>
	<h2>Plot</h2>
	<img src="data:image/png;base64,{{ plot_base64 }}" alt="forecast plot" />
</body>
</html>
"""


def plot_forecast(dates: List[pd.Timestamp], closes: List[float], forecast_dates: List[pd.Timestamp], forecast_values: List[float]) -> bytes:
	plt.figure(figsize=(10, 5))
	plt.plot(dates, closes, label="Historical Close")
	plt.plot(forecast_dates, forecast_values, label="Forecast", marker="o")
	plt.legend()
	plt.title("Close Price Forecast")
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.tight_layout()
	buf = io.BytesIO()
	plt.savefig(buf, format="png")
	plt.close()
	buf.seek(0)
	return buf.getvalue()


def render_html(ticker: str, generated_at: str, last_close: float, forecast_values: List[float], plot_base64: str) -> str:
	t = Template(HTML_TEMPLATE)
	return t.render(
		ticker=ticker,
		generated_at=generated_at,
		last_close=f"{last_close:.2f}",
		forecast_values=", ".join(f"{v:.2f}" for v in forecast_values),
		plot_base64=plot_base64,
	)