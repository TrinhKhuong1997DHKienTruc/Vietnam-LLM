from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TrainResult:
	history: keras.callbacks.History
	model: keras.Model


def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
	model = keras.Sequential([
		layers.Input(shape=input_shape),
		layers.LSTM(100, return_sequences=True),
		layers.Dropout(0.2),
		layers.LSTM(100, return_sequences=True),
		layers.Dropout(0.2),
		layers.LSTM(100, return_sequences=True),
		layers.Dropout(0.2),
		layers.LSTM(100),
		layers.Dropout(0.2),
		layers.Dense(1),
	])
	model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
	return model


def train(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 64) -> TrainResult:
	early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
	history = model.fit(
		x_train, y_train,
		validation_data=(x_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		verbose=2,
		callbacks=[early_stop],
	)
	return TrainResult(history=history, model=model)


def forecast_next_week(model: keras.Model, last_window: np.ndarray, steps: int = 5) -> np.ndarray:
	# last_window shape: (lookback, features)
	window = last_window.copy()
	preds = []
	for _ in range(steps):
		inp = window.reshape(1, window.shape[0], window.shape[1])
		yhat = model.predict(inp, verbose=0)[0, 0]
		preds.append(yhat)
		# shift window: append predicted close in the close index (3)
		next_row = window[-1].copy()
		next_row[3] = yhat
		window = np.vstack([window[1:], next_row])
	return np.array(preds)