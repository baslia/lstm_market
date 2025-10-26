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
import logging

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


# module-level logger
logger = logging.getLogger(__name__)


def configure_logging(ticker: str | None = None, verbose: bool = False):
    """Configure logging to console and optional per-ticker log file.

    If `ticker` is provided, a file `models/<ticker>/run.log` will be created and used.
    """
    level = logging.DEBUG if verbose else logging.INFO
    # reset root logger handlers to avoid duplicate logs on repeated calls
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if ticker:
        out_dir = os.path.join('models', ticker)
        os.makedirs(out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(out_dir, 'run.log'))
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def download_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download historical data (Open, Close) for `ticker` from Yahoo Finance."""
    logger.info("Downloading %s data for period=%s", ticker, period)
    df = yf.download(ticker, period=period, auto_adjust=True)
    if df.empty:
        logger.error("No data downloaded for ticker %s", ticker)
        raise ValueError(f"No data downloaded for ticker {ticker}")
    df = df[['Open', 'Close']].dropna()
    logger.debug("Downloaded %d rows for %s", len(df), ticker)
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
    logger.info("Starting training for %s (period=%s, seq_len=%d, epochs=%d, batch_size=%d)", ticker, period, seq_len, epochs, batch_size)
    df = download_data(ticker, period=period)
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df.values)
    logger.debug("Values shape after scaling: %s", values.shape)

    X, y = create_sequences(values, seq_len)
    logger.info("Created %d sequences (seq_len=%d)", len(X), seq_len)
    # If sklearn fallback, flatten sequences for non-recurrent model
    if not TF_AVAILABLE:
        logger.info("TensorFlow not available; using sklearn RandomForest fallback for %s", ticker)
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.1, shuffle=False)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = np.mean((val_pred - y_val) ** 2)
        logger.info("Fallback RandomForest validation MSE: %.6f", val_mse)

        out_dir = os.path.join('models', ticker)
        os.makedirs(out_dir, exist_ok=True)
        # compute residual statistics for CI
        # prepare inverse-transformed validation arrays and residuals; provide safe fallbacks
        val_actual = None
        val_pred_inv = None
        try:
            val_actual = scaler.inverse_transform(y_val)
            val_pred_inv = scaler.inverse_transform(val_pred)
            resid = val_actual - val_pred_inv
            resid_mean = np.mean(resid, axis=0)
            resid_std = np.std(resid, axis=0, ddof=1)
        except Exception:
            logger.exception('Could not compute residual statistics for RF fallback; attempting safe fallbacks')
            # Try to at least compute inverse transforms separately; otherwise fall back to zeros
            try:
                if val_actual is None:
                    val_actual = scaler.inverse_transform(y_val)
            except Exception:
                val_actual = np.zeros((len(y_val), values.shape[1]))
            try:
                if val_pred_inv is None:
                    val_pred_inv = scaler.inverse_transform(val_pred)
            except Exception:
                val_pred_inv = np.zeros_like(val_actual)
            try:
                resid = val_actual - val_pred_inv
                resid_mean = np.mean(resid, axis=0)
                resid_std = np.std(resid, axis=0, ddof=1)
            except Exception:
                resid_mean = np.zeros((values.shape[1],))
                resid_std = np.zeros((values.shape[1],))

        joblib.dump({'model': model, 'scaler': scaler, 'seq_len': seq_len, 'resid_mean': resid_mean, 'resid_std': resid_std}, os.path.join(out_dir, 'rf_model.joblib'))
        logger.info("Saved sklearn fallback model to %s", out_dir)

        # Plot validation predictions vs actual (if matplotlib available)
        if _HAS_MATPLOTLIB:
            try:
                actual = val_actual
                pred = val_pred_inv
                dates = pd.RangeIndex(start=0, stop=len(actual))

                plt.figure(figsize=(10, 6))
                plt.subplot(2, 1, 1)
                plt.plot(dates, actual[:, 0], label='Actual Open')
                plt.plot(dates, pred[:, 0], label='Predicted Open')
                plt.legend(); plt.title(f'{ticker} - Validation: Open (RF fallback)')

                plt.subplot(2, 1, 2)
                plt.plot(dates, actual[:, 1], label='Actual Close')
                plt.plot(dates, pred[:, 1], label='Predicted Close')
                plt.legend(); plt.title(f'{ticker} - Validation: Close (RF fallback)')

                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'val_predictions.png'))
                plt.close()
                logger.info("Saved validation plot to %s", os.path.join(out_dir, 'val_predictions.png'))
            except Exception:
                logger.exception("Could not create validation plot (RF fallback)")

        return

    # TensorFlow LSTM path
    logger.info("TensorFlow available; training LSTM model for %s", ticker)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
    logger.debug("Train/Val sizes: %d / %d", len(X_train), len(X_val))

    model = build_lstm_model((seq_len, values.shape[1]))
    es = EarlyStopping(patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es])

    out_dir = os.path.join('models', ticker)
    os.makedirs(out_dir, exist_ok=True)
    # save Keras model using recommended .keras extension
    model.save(os.path.join(out_dir, 'lstm_model.keras'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

    # compute residual statistics for CI and save to meta
    try:
        val_pred_scaled = model.predict(X_val)
        val_pred = scaler.inverse_transform(val_pred_scaled)
        val_actual = scaler.inverse_transform(y_val)
        resid = val_actual - val_pred
        resid_mean = np.mean(resid, axis=0)
        resid_std = np.std(resid, axis=0, ddof=1)
        joblib.dump({'resid_mean': resid_mean, 'resid_std': resid_std, 'seq_len': seq_len}, os.path.join(out_dir, 'meta.joblib'))
        logger.info("Saved residual statistics to %s", os.path.join(out_dir, 'meta.joblib'))
    except Exception:
        logger.exception('Could not compute and save residual statistics (TF)')

    logger.info("Saved Keras model and scaler to %s", out_dir)

    # plot training loss if matplotlib available
    if _HAS_MATPLOTLIB:
        try:
            plt.figure()
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.legend()
            plt.title(f'Training loss for {ticker}')
            plt.savefig(os.path.join(out_dir, 'loss.png'))
            plt.close()
            logger.info("Saved training loss plot to %s", os.path.join(out_dir, 'loss.png'))
        except Exception:
            logger.exception("Could not save training loss plot")

        # Additionally, create validation predictions vs actual plot
        try:
            val_pred_scaled = model.predict(X_val)
            val_pred = scaler.inverse_transform(val_pred_scaled)
            val_actual = scaler.inverse_transform(y_val)
            dates = pd.RangeIndex(start=0, stop=len(val_actual))

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(dates, val_actual[:, 0], label='Actual Open')
            plt.plot(dates, val_pred[:, 0], label='Predicted Open')
            plt.legend(); plt.title(f'{ticker} - Validation: Open')

            plt.subplot(2, 1, 2)
            plt.plot(dates, val_actual[:, 1], label='Actual Close')
            plt.plot(dates, val_pred[:, 1], label='Predicted Close')
            plt.legend(); plt.title(f'{ticker} - Validation: Close')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'val_predictions.png'))
            plt.close()
            logger.info("Saved validation predictions plot to %s", os.path.join(out_dir, 'val_predictions.png'))
        except Exception:
            logger.exception("Could not create validation predictions plot (TF)")

    logger.info("Finished training for %s", ticker)


def predict(ticker: str, period: str = '5y', seq_len: int = 20):
    out_dir = os.path.join('models', ticker)
    # check for tensorflow model first (support new .keras filename and legacy directory)
    keras_model_path = os.path.join(out_dir, 'lstm_model.keras')
    legacy_model_path = os.path.join(out_dir, 'lstm_model')
    scaler_path = os.path.join(out_dir, 'scaler.joblib')
    # prefer the .keras file but fall back to legacy directory if present
    if os.path.exists(keras_model_path):
        model_path = keras_model_path
    else:
        model_path = legacy_model_path

    # if TF model exists, use it
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        logger.info("Loading model from %s", model_path)
        df = download_data(ticker, period=period)
        scaler = joblib.load(scaler_path)
        values = scaler.transform(df.values)
        logger.debug("Transformed values shape for prediction: %s", values.shape)
        if len(values) < seq_len:
            logger.error('Not enough data to create a sequence')
            raise ValueError('Not enough data to create a sequence')
        last_seq = values[-seq_len:]
        model = tf.keras.models.load_model(model_path)
        pred_scaled = model.predict(last_seq.reshape(1, seq_len, -1))
        pred = scaler.inverse_transform(pred_scaled)
        pred_open, pred_close = pred[0]
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # create visualization: last seq actuals and predicted next-day
        if _HAS_MATPLOTLIB:
            try:
                recent = df[-seq_len:][['Open', 'Close']].copy()
                plot_df = recent.copy()
                plot_df.loc[next_date] = [pred_open, pred_close]

                plt.figure(figsize=(8, 5))
                plt.plot(plot_df.index, plot_df['Open'], marker='o', label='Open')
                plt.plot(plot_df.index, plot_df['Close'], marker='o', label='Close')
                plt.scatter([next_date], [pred_open], color='C0', s=100, marker='X')
                plt.scatter([next_date], [pred_close], color='C1', s=100, marker='X')
                plt.legend(); plt.title(f'{ticker} - Recent Open/Close and Predicted Next Day')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'prediction_plot.png'))
                plt.close()
                logger.info("Saved prediction plot to %s", os.path.join(out_dir, 'prediction_plot.png'))
            except Exception:
                logger.exception("Could not create prediction plot (TF)")

        logger.info("TF Prediction for %s on %s: Open=%.2f, Close=%.2f", ticker, next_date.date(), pred_open, pred_close)
        return {'date': str(next_date.date()), 'open': float(pred_open), 'close': float(pred_close)}

    # else check for sklearn fallback
    rf_path = os.path.join(out_dir, 'rf_model.joblib')
    if os.path.exists(rf_path):
        logger.info("Loading sklearn fallback model from %s", rf_path)
        df = download_data(ticker, period=period)
        meta = joblib.load(rf_path)
        model = meta['model']
        scaler = meta['scaler']
        seq_len = meta['seq_len']
        values = scaler.transform(df.values)
        logger.debug("Transformed values shape for prediction (RF): %s", values.shape)
        if len(values) < seq_len:
            logger.error('Not enough data to create a sequence')
            raise ValueError('Not enough data to create a sequence')
        last_seq = values[-seq_len:]
        pred_scaled = model.predict(last_seq.reshape(1, -1))
        pred = scaler.inverse_transform(pred_scaled)
        pred_open, pred_close = pred[0]
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)

        if _HAS_MATPLOTLIB:
            try:
                recent = df[-seq_len:][['Open', 'Close']].copy()
                plot_df = recent.copy()
                plot_df.loc[next_date] = [pred_open, pred_close]

                plt.figure(figsize=(8, 5))
                plt.plot(plot_df.index, plot_df['Open'], marker='o', label='Open')
                plt.plot(plot_df.index, plot_df['Close'], marker='o', label='Close')
                plt.scatter([next_date], [pred_open], color='C0', s=100, marker='X')
                plt.scatter([next_date], [pred_close], color='C1', s=100, marker='X')
                plt.legend(); plt.title(f'{ticker} - Recent Open/Close and Predicted Next Day (RF)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'prediction_plot.png'))
                plt.close()
                logger.info("Saved prediction plot to %s", os.path.join(out_dir, 'prediction_plot.png'))
            except Exception:
                logger.exception("Could not create prediction plot (RF)")

        logger.info("RF Fallback Prediction for %s on %s: Open=%.2f, Close=%.2f", ticker, next_date.date(), pred_open, pred_close)
        return {'date': str(next_date.date()), 'open': float(pred_open), 'close': float(pred_close)}

    logger.error("No trained model found for ticker %s. Run with --train first", ticker)
    raise FileNotFoundError('No trained model found for this ticker. Run with --train first')


# convenience function for programmatic use
def predict_next(ticker: str, train_if_missing: bool = False, **kwargs):
    """Predict next-day Open/Close for ticker.

    If no model exists and train_if_missing is True, trains a short model first.
    """
    out_dir = os.path.join('models', ticker)
    if not os.path.exists(out_dir) and train_if_missing:
        logger.info('No model found; training a short model as requested...')
        # quick train to produce a model
        train(ticker, epochs=5, seq_len=kwargs.get('seq_len', 20))
    return predict(ticker, seq_len=kwargs.get('seq_len', 20))


def forecast(ticker: str, days: int = 5, ci: float = 0.8, period: str = '5y', seq_len: int = 20):
    """Forecast `days` into the future using a recursive strategy and plot CI bands.

    Uses saved residual std (from validation set) to compute an approximate CI assuming normal errors.
    """
    out_dir = os.path.join('models', ticker)
    keras_model_path = os.path.join(out_dir, 'lstm_model.keras')
    rf_path = os.path.join(out_dir, 'rf_model.joblib')
    meta_path = os.path.join(out_dir, 'meta.joblib')

    # load data and scaler
    df = download_data(ticker, period=period)
    if os.path.exists(rf_path):
        meta = joblib.load(rf_path)
        model = meta['model']
        scaler = meta['scaler']
        saved_seq_len = meta.get('seq_len', seq_len)
        resid_std = meta.get('resid_std', None)
    elif os.path.exists(keras_model_path) and os.path.exists(os.path.join(out_dir, 'scaler.joblib')):
        model = tf.keras.models.load_model(keras_model_path)
        scaler = joblib.load(os.path.join(out_dir, 'scaler.joblib'))
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            resid_std = meta.get('resid_std', None)
            saved_seq_len = meta.get('seq_len', seq_len)
        else:
            resid_std = None
            saved_seq_len = seq_len
    else:
        logger.error('No model found to forecast for %s', ticker)
        raise FileNotFoundError('No trained model found for forecasting. Run --train first')

    # prepare sequence
    scaler_local = scaler
    values = scaler_local.transform(df.values)
    if len(values) < saved_seq_len:
        raise ValueError('Not enough historical data for the model seq_len')
    seq = values[-saved_seq_len:].copy()

    preds = []
    for day in range(days):
        if os.path.exists(rf_path):
            # flatten and predict
            x_in = seq.reshape(1, -1)
            pred_scaled = model.predict(x_in)
            # For RF, model.predict returns unscaled values (we saved scaler to inverse them below)
            # ensure shape (1,2)
        else:
            pred_scaled = model.predict(seq.reshape(1, saved_seq_len, -1))

        # if RF we saved predictions in original scale? In training we saved scaler, so model.predict returns scaled or unscaled?
        # Our RF was trained on scaled targets earlier, so rf.predict returns scaled-space targets. We'll inverse-transform using scaler.
        try:
            pred = scaler_local.inverse_transform(pred_scaled.reshape(1, -1))[0]
        except Exception:
            # fallback: if model already predicts in original scale
            pred = np.array(pred_scaled).reshape(-1)[:2]

        preds.append(pred)

        # append predicted scaled value to seq for next iteration
        try:
            pred_scaled_for_seq = scaler_local.transform(pred.reshape(1, -1))[0]
        except Exception:
            pred_scaled_for_seq = pred_scaled.reshape(-1)[:seq.shape[1]]

        seq = np.vstack([seq[1:], pred_scaled_for_seq])

    preds = np.array(preds)  # shape (days, 2)

    # compute CI using resid_std (per-feature). If missing, estimate from recent historical changes
    if resid_std is None:
        # estimate residual std from recent returns (difference of last 30 days)
        recent = df[['Open', 'Close']].values[-30:]
        deltas = np.diff(recent, axis=0)
        est_std = np.std(deltas, axis=0, ddof=1)
        resid_std = est_std
        logger.warning('Residual std not found in metadata; using estimated std from recent deltas: %s', resid_std)

    # z for two-sided CI
    z = abs(np.percentile(np.random.normal(size=100000), [(1+ci)/2*100]))[0]
    # better to compute z directly using scipy or known value, but to avoid extra dep: use inverse of normal CDF approx
    # Replace with exact value for common CI (80% -> z ~ 1.28155)
    if abs(ci - 0.8) < 1e-6:
        z = 1.281551565545

    lower = preds - z * resid_std
    upper = preds + z * resid_std

    # prepare plot: recent history + forecast
    if _HAS_MATPLOTLIB:
        try:
            # show only the last N days of historical data before the forecast to keep the plot readable
            N = 30
            recent = df[['Open', 'Close']].copy()
            if len(recent) > N:
                recent = recent.tail(N)
            last_index = recent.index[-1]
            future_index = [last_index + pd.Timedelta(days=i+1) for i in range(days)]

            plt.figure(figsize=(10, 6))
            # Open
            plt.subplot(2, 1, 1)
            plt.plot(recent.index, recent['Open'], label='Open (historical)')
            plt.plot(future_index, preds[:, 0], marker='o', linestyle='--', label='Predicted Open')
            plt.fill_between(future_index, lower[:, 0], upper[:, 0], color='C0', alpha=0.2, label=f'{int(ci*100)}% CI')
            plt.legend(); plt.title(f'{ticker} - Forecast Open ({days} days)')

            # Close
            plt.subplot(2, 1, 2)
            plt.plot(recent.index, recent['Close'], label='Close (historical)')
            plt.plot(future_index, preds[:, 1], marker='o', linestyle='--', label='Predicted Close')
            plt.fill_between(future_index, lower[:, 1], upper[:, 1], color='C1', alpha=0.2, label=f'{int(ci*100)}% CI')
            plt.legend(); plt.title(f'{ticker} - Forecast Close ({days} days)')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'forecast_plot.png'))
            plt.close()
            logger.info('Saved forecast plot to %s', os.path.join(out_dir, 'forecast_plot.png'))
        except Exception:
            logger.exception('Could not create forecast plot')

    # return structured forecast
    out = []
    for i in range(days):
        out.append({'date': str((df.index[-1] + pd.Timedelta(days=i+1)).date()),
                    'open': float(preds[i, 0]),
                    'open_lower': float(lower[i, 0]),
                    'open_upper': float(upper[i, 0]),
                    'close': float(preds[i, 1]),
                    'close_lower': float(lower[i, 1]),
                    'close_upper': float(upper[i, 1])})

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--period', default='5y')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--forecast', action='store_true', help='Forecast N days into the future and save plot')
    parser.add_argument('--forecast_days', type=int, default=5, help='Number of days to forecast')
    parser.add_argument('--ci', type=float, default=0.8, help='Confidence level for forecast intervals (0-1)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose (DEBUG) logging')
    args = parser.parse_args()

    # configure logging (creates models/<ticker>/run.log when ticker provided)
    configure_logging(args.ticker, verbose=args.verbose)

    if args.train:
        train(args.ticker, period=args.period, seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size)
    if args.predict:
        predict(args.ticker, period=args.period, seq_len=args.seq_len)
    if args.forecast:
        fc = forecast(args.ticker, days=args.forecast_days, ci=args.ci, period=args.period, seq_len=args.seq_len)
        logger.info('Forecast results: %s', fc)


if __name__ == '__main__':
    main()
