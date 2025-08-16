import os
import datetime as dt
import yfinance as yf


def generate_report(ticker, output_dir="reports_simple"):
    os.makedirs(output_dir, exist_ok=True)

    end = dt.date.today()
    start = end - dt.timedelta(days=90)

    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        print(f"No data for {ticker}")
        return

    # compute indicators
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA5"] = data["Close"].rolling(window=5).mean()

    current_price = float(data["Close"].iloc[-1])
    ma20 = float(data["MA20"].iloc[-1])
    ma5 = float(data["MA5"].iloc[-1])

    last_week_change = float((data["Close"].iloc[-1] - data["Close"].iloc[-5]) / data["Close"].iloc[-5] * 100)

    # simple rule-based forecast
    if current_price > ma20 and ma5 > ma20:
        forecast_direction = "up"
        forecast_pct = "+2% to +4%"
        reasoning = "Price above 20-day moving average with short-term momentum positive."
    elif current_price < ma20 and ma5 < ma20:
        forecast_direction = "down"
        forecast_pct = "-2% to -4%"
        reasoning = "Price below 20-day moving average with short-term momentum negative."
    else:
        forecast_direction = "sideways"
        forecast_pct = "-1% to +1%"
        reasoning = "Mixed signals between short-term and medium-term trends."

    report_lines = [
        f"# Market Forecast Report: {ticker}",
        f"Date: {end}",
        "",
        f"Current Price: ${current_price:.2f}",
        f"20-day MA: ${ma20:.2f}",
        f"5-day MA: ${ma5:.2f}",
        f"Last Week Change: {last_week_change:.2f}%", 
        "", 
        "## Forecast", 
        f"Expected Direction for next week: **{forecast_direction}** ({forecast_pct})", 
        "", 
        "### Rationale", 
        reasoning, 
    ]

    filepath = os.path.join(output_dir, f"{ticker}_forecast.md")
    with open(filepath, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved simple forecast for {ticker} to {filepath}")


if __name__ == "__main__":
    for t in ["MSFT", "NVDA"]:
        generate_report(t)