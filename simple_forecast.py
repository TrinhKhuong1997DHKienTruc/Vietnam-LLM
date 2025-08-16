import os
from datetime import datetime, timedelta
import yfinance as yf


def generate_simple_forecast(ticker: str, horizon_days: int = 7, output_dir: str = "reports"):
    """Generate a naive price trend forecast based on recent momentum."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Compute recent returns
    data['Return'] = data['Close'].pct_change()
    recent_return = data['Return'][-5:].mean()

    # Forecast direction
    direction = "up" if recent_return > 0 else "down"
    pct = abs(recent_return) * 100 * 5  # scale naive estimation
    forecast_str = f"{direction} by {pct:.2f}%"

    # Compose report
    report = [
        f"Ticker: {ticker}",
        f"Date Generated: {end_date.strftime('%Y-%m-%d')}",
        "", 
        "Recent Performance:",
        f" - 1 Week return: {data['Return'][-5:].sum()*100:.2f}%",
        f" - 1 Month return: {data['Return'].sum()*100:.2f}%",
        "", 
        f"Forecast for next week: {forecast_str}",
        "", 
        "Rationale:",
        "This is a simple momentum-based forecast using the average return of the past week. ",
        "No sophisticated machine learning or fundamental analysis is applied. Use with caution."
    ]

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"simple_forecast_{ticker}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Saved simple forecast for {ticker}: {path}")


if __name__ == "__main__":
    for sym in ["MSFT", "NVDA"]:
        generate_simple_forecast(sym)