import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from chronos import ChronosPipeline
except Exception as e:
    print("Failed to import Chronos. Make sure requirements are installed. Error:", e)
    raise


def infer_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def load_aapl_data() -> pd.DataFrame:
    """Load AAPL OHLCV using yfinance (no API keys required)."""
    import yfinance as yf

    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * 3)
    df = yf.download("AAPL", start=start.isoformat(), end=end.isoformat(), progress=False)
    if df is None or df.empty:
        raise RuntimeError("Failed to download AAPL data via yfinance.")
    df = df.reset_index()
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def prepare_series(df: pd.DataFrame, column: str = "Close") -> tuple[np.ndarray, pd.DatetimeIndex]:
    series = df[column].astype(float).to_numpy()
    dates = pd.to_datetime(df["date"])  # type: ignore[arg-type]
    return series, dates


def forecast_next_7_days(series: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    # Use last N points as context. Chronos handles scaling internally.
    context_length = min(256, max(32, series.shape[0] - 7))
    context = torch.tensor(series[-context_length:], dtype=torch.float32, device=device)

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map=str(device),
        torch_dtype=dtype,
    )

    prediction_length = 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = pipeline.predict(context, prediction_length=prediction_length, num_samples=100)

    # forecast is typically [num_samples, prediction_length]
    if isinstance(forecast, torch.Tensor):
        samples = forecast.detach().cpu().numpy()
    else:
        # Try to coerce if the pipeline returns numpy already
        samples = np.asarray(forecast)

    mean_forecast = samples.mean(axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    return np.stack([p10, mean_forecast, p90], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-out", type=str, default="chronos-bolt-aapl/assets/aapl_forecast.png")
    parser.add_argument("--price-col", type=str, default="Close", help="Which price column to forecast")
    args = parser.parse_args()

    device, dtype = infer_device_and_dtype()
    df = load_aapl_data()
    series, dates = prepare_series(df, column=args.price_col)

    quantiles = forecast_next_7_days(series, device, dtype)
    p10, mean_fc, p90 = quantiles

    # Build dates for forecast horizon (next calendar days based on last date)
    last_date = dates.iloc[-1]
    horizon = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq="D")

    # Plot
    os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.plot(dates[-200:], series[-200:], label="History", color="#1f77b4")
    plt.plot(horizon, mean_fc, label="Chronos-Bolt mean", color="#d62728")
    plt.fill_between(horizon, p10, p90, color="#ff9896", alpha=0.35, label="P10�P90")
    plt.title("AAPL 7-day Forecast (amazon/chronos-bolt-base)")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({args.price_col})")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150)
    print(f"Saved plot to: {args.plot_out}")


if __name__ == "__main__":
    main()