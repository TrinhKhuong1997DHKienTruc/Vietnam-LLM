import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import yfinance as yf

SEED = 42
np.random.seed(SEED)

def download_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, threads=False)
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}.")
    df = df[['Close']].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def prepare_data(series: pd.Series, look_back: int = 60):
    scaler = None
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_model(input_shape):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    tf.random.set_seed(SEED)

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast(model, last_sequence, steps, scaler):
    preds = []
    seq = last_sequence.copy()
    for _ in range(steps):
        pred = model.predict(seq[np.newaxis, :, :], verbose=0)[0][0]
        preds.append(pred)
        seq = np.append(seq[1:], [[pred]], axis=0)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


def generate_report(ticker: str, predictions: np.ndarray):
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    today = dt.date.today()
    next_dates = pd.bdate_range(today + dt.timedelta(days=1), periods=len(predictions)).date
    lines = [f"# {ticker} Stock Price Forecast\n", f"Generated on {today}\n", "\n", "| Date | Predicted Close |", "|---|---|"]
    for d, p in zip(next_dates, predictions):
        lines.append(f"| {d} | {p:.2f} |")
    content = "\n".join(lines)
    out_path = report_dir / f"{ticker}_forecast_{today}.md"
    out_path.write_text(content)
    print(f"Report written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate next-week forecast for a stock.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g., MSFT")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    df = download_data(args.ticker)
    look_back = 60
    X, y, scaler = prepare_data(df['Close'], look_back)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=32, verbose=2)

    last_sequence = X[-1]
    predictions = forecast(model, last_sequence, 5, scaler)  # Next 5 business days

    generate_report(args.ticker, predictions)

    # Save model if needed
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / f"lstm_{args.ticker}.keras")

if __name__ == "__main__":
    main()