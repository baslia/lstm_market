# LSTM Stock Predictor

A small example project that downloads historical stock Open/Close prices from Yahoo Finance and trains a model to predict the next day's Open and Close prices.

The project prefers to use an LSTM implemented with TensorFlow. If TensorFlow is not available in your environment, the script falls back to a scikit-learn RandomForestRegressor so you can run quick smoke tests without installing large packages. The fallback is provided for convenience and is not a substitute for the LSTM model.

Contents
- `src/predict_lstm.py` — Main script and library functions (training, prediction, programmatic helper).
- `requirements.txt` — Python dependencies you can install with pip.
- `models/` — (created after training) saved models and scalers per-ticker.

Quick start

1. Create and activate a virtual environment (macOS / Linux example):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- If you don't want to install TensorFlow, the script will still run and use a RandomForest fallback; install TensorFlow if you want the LSTM model (training with TF is more computationally intensive).

Train a model (CLI)

Train a model for a ticker (example: AAPL) and save it under `models/AAPL/`:

```bash
python src/predict_lstm.py --ticker AAPL --train
```

Optional flags:
- `--period` default `5y` — history period for yfinance (e.g. `1y`, `2y`, `5y`).
- `--seq_len` default `20` — sequence length (how many past days used for prediction).
- `--epochs` default `50` — number of epochs (only used when TensorFlow is available).
- `--batch_size` default `32` — batch size (only used when TensorFlow is available).

Predict next-day Open/Close (CLI)

Assuming you've trained previously and models are saved in `models/<TICKER>/`:

```bash
python src/predict_lstm.py --ticker AAPL --predict
```

Examples

Train and immediately predict:

```bash
python src/predict_lstm.py --ticker MSFT --train
python src/predict_lstm.py --ticker MSFT --predict
```

Programmatic usage

You can import the helper function for use inside other Python code. Example:

```python
from src.predict_lstm import predict_next

# Will train a very short model if no model exists when train_if_missing=True
result = predict_next('AAPL', train_if_missing=True)
print(result)  # {'date': 'YYYY-MM-DD', 'open': 123.45, 'close': 125.67}
```

Where models are saved

After training you'll find a folder `models/<TICKER>/`.
- TensorFlow LSTM path: `models/<TICKER>/lstm_model/` (saved Keras model folder) and `models/<TICKER>/scaler.joblib` for the MinMax scaler.
- Sklearn fallback: `models/<TICKER>/rf_model.joblib` containing a dict with keys `model`, `scaler`, and `seq_len`.

Notes, caveats and troubleshooting

- This is an educational example, not financial advice. Use predictions with caution.
- The model uses only the historical Open and Close prices (no volume, indicators or macro data). For production use you should add more features, cross-validation, hyperparameter tuning and rigorous backtesting.
- If you see errors like `No module named 'yfinance'` or `No module named 'tensorflow'`, install the missing packages via `pip install -r requirements.txt`.
- Example: to force the fallback behavior, run without TensorFlow installed; to use LSTM, install TensorFlow (note: TF can be large and requires additional system resources).

License

This project is provided as-is for educational purposes.

Contact

If you want changes (more features, indicators, or a Jupyter notebook example), tell me which ticker and what behavior you want and I can add it.
