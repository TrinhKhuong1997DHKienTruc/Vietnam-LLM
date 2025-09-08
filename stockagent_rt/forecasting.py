from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastResult:
	future_dates: List[pd.Timestamp]
	predicted_close: List[float]
	last_close: float


def _safe_series(values: pd.Series) -> pd.Series:
	s = pd.to_numeric(values, errors="coerce").dropna()
	return s.astype(float)


def holt_winters_forecast(close_series: pd.Series, periods: int = 14) -> ForecastResult:
	s = _safe_series(close_series)
	if len(s) < 10:
		raise ValueError("Not enough data for Holt-Winters")
	model = ExponentialSmoothing(s, trend="add", seasonal=None)
	fitted = model.fit(optimized=True)
	forecast_vals = fitted.forecast(periods)
	last_date = close_series.index[-1]
	future_dates = pd.bdate_range(start=last_date, periods=periods + 1, inclusive="neither")
	return ForecastResult(
		future_dates=list(future_dates[:periods]),
		predicted_close=list(np.maximum(0.0, forecast_vals.values)),
		last_close=float(s.iloc[-1]),
	)


def arima_forecast(close_series: pd.Series, periods: int = 14) -> ForecastResult:
	s = _safe_series(close_series)
	if len(s) < 20:
		raise ValueError("Not enough data for ARIMA")
	# Simple ARIMA fallback
	model = ARIMA(s, order=(1, 1, 1))
	fitted = model.fit()
	forecast_vals = fitted.forecast(steps=periods)
	last_date = close_series.index[-1]
	future_dates = pd.bdate_range(start=last_date, periods=periods + 1, inclusive="neither")
	return ForecastResult(
		future_dates=list(future_dates[:periods]),
		predicted_close=list(np.maximum(0.0, forecast_vals.values)),
		last_close=float(s.iloc[-1]),
	)


def forecast_next_14_days(close_series: pd.Series) -> ForecastResult:
	try:
		return holt_winters_forecast(close_series, periods=14)
	except Exception:
		return arima_forecast(close_series, periods=14)
