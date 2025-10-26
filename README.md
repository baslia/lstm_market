# LSTM Stock Predictor

A small example project that downloads historical stock Open/Close prices from Yahoo Finance and trains a model to predict the next day's Open and Close prices.

This repository provides a small, runnable script and helper functions to:
- Download Open/Close price history from Yahoo Finance.
- Train a TensorFlow Keras LSTM to predict next-day Open & Close (multivariate).
- Fall back to a scikit-learn RandomForestRegressor when TensorFlow is not installed (fast smoke tests).
- Produce visualizations for training, validation, single-step prediction, and multi-day forecasts with confidence intervals.
- Save models, scalers, metadata and per-run logs under `models/<TICKER>/`.

Important note

This is an educational example — not financial advice. Use predictions and forecasts with caution and do proper backtesting before using any model in production.

Contents
- `src/predict_lstm.py` — Main script and library functions (train, predict, forecast, programmatic helpers).
- `requirements.txt` — Suggested Python dependencies.
- `models/` — Created after training; stores models, scalers, plots, and logs per ticker.

Quick start

1) Create and activate a virtual environment (macOS / Linux example):

```bash
python -m venv venv
source venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

Notes about dependencies
- `tensorflow` is required for the LSTM path. Installing TF may be large and take time. If `tensorflow` is not installed, the script uses a RandomForest fallback (`scikit-learn`) so you can test functionality quickly.
- `matplotlib` is optional but recommended if you want plots saved to disk.

Basic workflow (CLI)

Train a model for `AAPL` (default 5 years history):

```bash
python src/predict_lstm.py --ticker AAPL --train
```

Predict the next-day Open/Close using the saved model:

```bash
python src/predict_lstm.py --ticker AAPL --predict
```

Forecast N days into the future with an 80% confidence interval (default 5 days):

```bash
python src/predict_lstm.py --ticker AAPL --forecast --forecast_days 7 --ci 0.8
```

Enable verbose logging (prints DEBUG and also saves to `models/<TICKER>/run.log`):

```bash
python src/predict_lstm.py --ticker AAPL --train --verbose
```

Hyperparameter tuning (new)

The script supports a simple randomized hyperparameter tuning flow for the LSTM model to quickly search a small space of architectures and learning rates. Tuning only runs when TensorFlow is available and is enabled with the `--tune` flag.

- CLI example (tune then train):

```bash
python src/predict_lstm.py --ticker AAPL --train --tune --tune_trials 12 --verbose
```

- Flags:
  - `--tune` — enable randomized hyperparameter tuning.
  - `--tune_trials` — number of randomized trials to run (default 12).

- What the tuner searches (small randomized space):
  - LSTM units: [32, 64, 128]
  - Dropout: [0.1, 0.2, 0.3]
  - Dense units: [16, 32, 64]
  - Learning rate: [1e-3, 5e-4, 1e-4]
  - Batch size: [32, 64]

- Behavior:
  - For each trial the tuner trains a small model (short epochs + EarlyStopping) and evaluates validation loss.
  - The best hyperparameters are saved to `models/<TICKER>/tune_best_params.joblib`.
  - After tuning, the final model is built using the best parameters and trained for the full number of epochs you specified.

- Caveats:
  - The tuner is intentionally simple and quick (randomized trials). For more thorough tuning consider KerasTuner or Optuna.
  - Tuning will be slower and consume more compute; decrease `--tune_trials` when testing locally or use fewer `--epochs` during tuning.

Key CLI flags
- `--ticker` (required): ticker symbol to download/train/predict for (e.g. AAPL).
- `--train`: train a model and save artifacts to `models/<TICKER>/`.
- `--predict`: load a trained model and predict next-day Open/Close.
- `--forecast`: produce a multi-day forecast and save a forecast plot.
- `--forecast_days` (default 5): number of future days to forecast when using `--forecast`.
- `--ci` (default 0.8): confidence level for forecast intervals (0 < ci < 1).
- `--period` (default `5y`): history period passed to `yfinance` (e.g. `1y`, `2y`, `5y`).
- `--seq_len` (default 20): number of past days used as input to the model.
- `--epochs`, `--batch_size`: training hyperparams (used only when TensorFlow is available).
- `--tune`: enable hyperparameter tuning for LSTM.
- `--tune_trials`: number of tuning trials to run when `--tune` is used.
- `--verbose`: enable DEBUG logging and create `models/<TICKER>/run.log`.

What the script saves (models/<TICKER>/)
- `lstm_model.keras` — saved Keras model (if TensorFlow path used).
- `scaler.joblib` — MinMaxScaler used to scale inputs/outputs.
- `meta.joblib` — metadata including validation residuals (`resid_mean`, `resid_std`) and `seq_len` (TF path).
- `rf_model.joblib` — sklearn fallback artifact (contains `model`, `scaler`, `seq_len`, and residual stats).
- `tune_best_params.joblib` — best hyperparameters discovered during tuning (if `--tune` used).
- `loss.png` — training & validation loss curve (TF path, if matplotlib available).
- `val_predictions.png` — validation set actual vs predicted plot (TF or RF path if matplotlib available).
- `prediction_plot.png` — recent days + single-step predicted next day (created when running `--predict`).
- `forecast_plot.png` — multi-day forecast plot showing predicted points with shaded CI bands (created when `--forecast` is used).
- `run.log` — (created if `--verbose` provided) per-run log with INFO/DEBUG messages.

Forecasting and confidence intervals

- The `--forecast` action performs recursive (autoregressive) multi-day forecasting: each predicted day is appended to the input sequence to predict the next step.
- Forecast CI bands: the script stores validation residual statistics during training (per-feature residual standard deviation). Forecast intervals are computed as prediction ± z * resid_std assuming approximately normal residuals. For 80% CI the z score used is ≈ 1.28155.
- If residual statistics are missing, the script estimates a suitable scale from recent historical day-to-day deltas as a fallback.
- Caveat: these intervals are approximate. For more robust uncertainty estimates consider bootstrapping or probabilistic models.

Programmatic usage

You can import the functions directly:

```python
from src.predict_lstm import train, predict, forecast, predict_next, configure_logging

# enable logging into models/AAPL/run.log
configure_logging('AAPL', verbose=True)
train('AAPL', epochs=5)  # short train
res = predict('AAPL')
fc = forecast('AAPL', days=7, ci=0.8)
print(res)
print(fc)
```

Interpreting the forecast output

The `forecast(...)` returns a list of dictionaries, one per future day:

{
  'date': 'YYYY-MM-DD',
  'open': float,            # predicted Open
  'open_lower': float,      # lower bound of CI for Open
  'open_upper': float,      # upper bound of CI for Open
  'close': float,           # predicted Close
  'close_lower': float,     # lower bound of CI for Close
  'close_upper': float      # upper bound of CI for Close
}

Troubleshooting

- Missing packages errors (e.g., "No module named 'yfinance'") — install dependencies:

```bash
pip install -r requirements.txt
```

- If you want LSTM training/prediction, ensure `tensorflow` is installed in your environment. If not installed, the script uses a RandomForest fallback.
- If no plots are produced, ensure `matplotlib` is installed. The script will continue to work without plotting but will log the missing plotting actions.
- The script assumes calendar days when adding future dates. If you prefer business-day forecasting (skip weekends/holidays), I can change the forecast x-axis to use business days.

Extending this project

- Add volume and technical indicators as inputs (improves feature richness).
- Add hyperparameter search and k-fold time-series cross-validation.
- Replace simple CI with bootstrap-based prediction intervals or use probabilistic forecasting (e.g., Bayesian RNN, Monte Carlo dropout).
- Add a Jupyter notebook that visualizes training, predictions and forecasts interactively.

License

This project is provided as-is for educational purposes.

Contact

If you'd like me to run a quick RF-based training and forecast for a ticker and attach the plot, tell me which ticker and I will run it and show the results.
