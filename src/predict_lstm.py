#!/usr/bin/env python3
"""
Simple LSTM predictor for stock Open/Close using yfinance.

Usage:
    python src/predict_lstm.py --ticker AAPL --train
    python src/predict_lstm.py --ticker AAPL --predict

This script prefers to use TensorFlow (LSTM). If TensorFlow is not available
it will fall back to a scikit-learn RandomForestRegressor (non-LSTM) so you can
run quick smoke tests without installing large packages. The fallback is only
for convenience and does not use an LSTM.
"""

import argparse
import os
import joblib

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# Try to import TensorFlow; if unavailable, provide a sklearn fallback model
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    TF_AVAILABLE = False
    tf = None

from sklearn.ensemble import RandomForestRegressor


def download_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download historical data (Open, Close) for `ticker` from Yahoo Finance."""
    print(f"Downloading {ticker} data for period={period}...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}")
    df = df[['Open', 'Close']].dropna()
    return df


def create_sequences(values: np.ndarray, seq_len: int):
    """Create sequences for time-series modelling.

    Returns X, y where X shape is (n_samples, seq_len, n_features) and y is (n_samples, n_features).
    """
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model


def train(ticker: str, period: str = '5y', seq_len: int = 20, epochs: int = 50, batch_size: int = 32):
    df = download_data(ticker, period=period)
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df.values)

    X, y = create_sequences(values, seq_len)
    # If sklearn fallback, flatten sequences for non-recurrent model
    if not TF_AVAILABLE:
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.1, shuffle=False)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = np.mean((val_pred - y_val) ** 2)
        print(f"Fallback RandomForest validation MSE: {val_mse:.6f}")

        out_dir = os.path.join('models', ticker)
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler, 'seq_len': seq_len}, os.path.join(out_dir, 'rf_model.joblib'))
        print(f"Saved sklearn fallback model to {out_dir}")
        return

    # TensorFlow LSTM path
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    model = build_lstm_model((seq_len, values.shape[1]))
    es = EarlyStopping(patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es])

    out_dir = os.path.join('models', ticker)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, 'lstm_model'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

    # plot training loss if matplotlib available
    if _HAS_MATPLOTLIB:
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.title(f'Training loss for {ticker}')
        plt.savefig(os.path.join(out_dir, 'loss.png'))

    print(f"Saved model and scaler to {out_dir}")


def predict(ticker: str, period: str = '5y', seq_len: int = 20):
    out_dir = os.path.join('models', ticker)
    # check for tensorflow model first
    model_path = os.path.join(out_dir, 'lstm_model')
    scaler_path = os.path.join(out_dir, 'scaler.joblib')

    # if TF model exists, use it
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        df = download_data(ticker, period=period)
        scaler = joblib.load(scaler_path)
        values = scaler.transform(df.values)
        if len(values) < seq_len:
            raise ValueError('Not enough data to create a sequence')
        last_seq = values[-seq_len:]
        model = tf.keras.models.load_model(model_path)
        pred_scaled = model.predict(last_seq.reshape(1, seq_len, -1))
        pred = scaler.inverse_transform(pred_scaled)
        pred_open, pred_close = pred[0]
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        print(f"TF Prediction for {ticker} on {next_date.date()}: Open={pred_open:.2f}, Close={pred_close:.2f}")
        return {'date': str(next_date.date()), 'open': float(pred_open), 'close': float(pred_close)}

    # else check for sklearn fallback
    rf_path = os.path.join(out_dir, 'rf_model.joblib')
    if os.path.exists(rf_path):
        df = download_data(ticker, period=period)
        meta = joblib.load(rf_path)
        model = meta['model']
        scaler = meta['scaler']
        seq_len = meta['seq_len']
        values = scaler.transform(df.values)
        if len(values) < seq_len:
            raise ValueError('Not enough data to create a sequence')
        last_seq = values[-seq_len:]
        pred_scaled = model.predict(last_seq.reshape(1, -1))
        pred = scaler.inverse_transform(pred_scaled)
        pred_open, pred_close = pred[0]
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        print(f"RF Fallback Prediction for {ticker} on {next_date.date()}: Open={pred_open:.2f}, Close={pred_close:.2f}")
        return {'date': str(next_date.date()), 'open': float(pred_open), 'close': float(pred_close)}

    raise FileNotFoundError('No trained model found for this ticker. Run with --train first')


# convenience function for programmatic use
def predict_next(ticker: str, train_if_missing: bool = False, **kwargs):
    """Predict next-day Open/Close for ticker.

    If no model exists and train_if_missing is True, trains a short model first.
    """
    out_dir = os.path.join('models', ticker)
    if not os.path.exists(out_dir) and train_if_missing:
        print('No model found; training a short model as requested...')
        # quick train to produce a model
        train(ticker, epochs=5, seq_len=kwargs.get('seq_len', 20))
    return predict(ticker, seq_len=kwargs.get('seq_len', 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--period', default='5y')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    if args.train:
        train(args.ticker, period=args.period, seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size)
    if args.predict:
        predict(args.ticker, period=args.period, seq_len=args.seq_len)


if __name__ == '__main__':
    main()
