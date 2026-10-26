# Stockformer 2.0: Transformer-Based Financial Time-Series Forecasting

A machine learning project that predicts the **next-hour return** of a stock (default: `ICICIBANK.NS`) from the previous 60 hours of price data and technical indicators. Two deep learning models, **Stockformer** (CNN + Transformer encoder) and **PatchTST**, are trained with a direction-aware loss, compared on an unseen test set, and visualized in an interactive **Streamlit** dashboard.

> Course project (ML Project 25RK28). Built for learning and experimentation, **not** for real trading.

---

## Features

- Hourly data download from Yahoo Finance (`yfinance`), last 730 days
- Feature engineering: log return, RSI, MACD, ATR, volume moving average
- **Stockformer**: 1D-CNN embedding + positional encoding + Transformer encoder
- **PatchTST**: patch-based Transformer from the `tsai` library (used as a comparison model)
- **Asymmetric MSE loss**: wrong-direction predictions are penalized 2.5x more
- Backtest-style evaluation: MSE, MAE, ROI, Sharpe ratio, compared with Buy & Hold
- Streamlit dashboard with four pages

---

## Project Structure

```
ML_finanacial_series/
├── app.py                  # Streamlit dashboard
├── index.py                # Plots training vs validation loss (matplotlib)
├── requirements.txt        # Python dependencies
├── data/
│   └── processed/          # features.csv, target.csv, X_test.npy, y_test.csv
├── models/                 # Saved weights (.pth) and scaler (.joblib)
├── notebooks/
│   └── TR_1_0.ipynb        # Baseline: Granger causality + first Stockformer
├── reports/                # loss_history.json, test_results.csv
└── src/
    ├── data/
    │   ├── build_features.py   # Download data and engineer features
    │   └── make_dataset.py     # Save processed features and target
    └── models/
        ├── model.py            # Stockformer + loss functions
        ├── train.py            # Training pipeline
        └── predict.py          # Inference and evaluation
```

---

## How It Works

1. **Data**: `build_features.py` downloads hourly OHLCV data for the chosen ticker.
2. **Features (12 total)**: Open, High, Low, Close, Volume, intra-hour log return `ln(Close/Open)`, RSI(14), MACD (3 columns), ATR(14), 20-hour volume SMA.
3. **Target**: the *next* hour's log return (shifted by one step, so there is no look-ahead leakage).
4. **Windows**: 60-hour lookback sequences, split chronologically (no shuffling): 80% train/validation, 20% test, with 15% of the training part used for validation. The scaler is fit on training data only.
5. **Training**: both models use the same asymmetric loss, Adam optimizer, and a `ReduceLROnPlateau` scheduler. The best checkpoint (lowest validation loss) is saved.
6. **Evaluation**: predictions act as position sizes. Strategy return per hour = `Actual return x Predicted return`. ROI is the compounded total, and Sharpe is mean divided by standard deviation of strategy returns.

### Models

| Model | Description |
|---|---|
| **Stockformer** | Conv1D (kernel 3) to 128-dim embeddings, positional encoding, 3 Transformer encoder layers, 8 attention heads, linear output from the last time step |
| **PatchTST** | `tsai` implementation, patch length 10, stride 5, RevIN normalization, residual attention |

### Loss function

`AsymmetricMSELoss(alpha=2.5)`: standard squared error, multiplied by `alpha` whenever the predicted sign differs from the actual sign. A `SharpeRatioLoss` is also defined in `model.py` for experimentation.

---

## Installation

```bash
git clone https://github.com/Maruthi5760/ML_finanacial_series.git
cd ML_finanacial_series

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install tsai
```

Python 3.9 or newer is recommended. `tsai` is required by `train.py` and `predict.py` for PatchTST.

---

## Usage

Run these from the project root folder, in order:

```bash
# 1. Download data and build features
python src/data/make_dataset.py

# 2. Train both models (saves weights, scaler, test data, loss history)
python src/models/train.py

# 3. Run inference and evaluation (saves reports/test_results.csv)
python src/models/predict.py

# 4. Launch the dashboard
streamlit run app.py
```

Generated files (`data/processed/`, `models/`, `reports/`) are recreated by the scripts above.

---

## Dashboard Pages

- **Project Overview**: summary of the project and method
- **Data Exploration**: enter any ticker, view raw data, closing price, RSI and engineered features
- **Training Curves**: training vs validation loss for both models
- **Model Results**: predicted vs actual returns, cumulative ROI vs Buy & Hold, metrics table, ROI vs Sharpe plot

---

## Configuration

Main settings are in `src/models/train.py`:

| Parameter | Value |
|---|---|
| `SEQ_LENGTH` | 60 hours |
| `BATCH_SIZE` | 64 |
| `LEARNING_RATE` | 0.01 |
| `NUM_EPOCHS` | 5 (quick test; increase for a real run) |
| `AsymmetricMSELoss alpha` | 2.5 |

To change the stock or data window, edit the ticker, period and interval in `src/data/make_dataset.py`.

---

## Current Status and Limitations

This is a work in progress, and the current results should be treated as a baseline:

- The models were trained for only 5 epochs with a high learning rate. On the current test set, neither model shows meaningful predictive skill (direction accuracy is close to 50%), and the Stockformer output is nearly constant.
- Any positive ROI should be interpreted carefully, since it can reflect market direction during the test period rather than forecasting ability.
- Sharpe ratio is calculated per hour and is not annualized. Transaction costs and slippage are not modeled.
- The overview mentions VMD (Variational Mode Decomposition), but it is **not yet implemented** in the code. It is planned future work.

### Planned improvements

- Train longer with a lower learning rate and a scaled target
- Implement VMD decomposition before PatchTST
- Use stationary features (returns instead of raw prices)
- Add transaction costs and annualized metrics to the backtest

---

## Tech Stack

Python, PyTorch, tsai (PatchTST), scikit-learn, pandas, NumPy, pandas_ta, yfinance, statsmodels, Streamlit, Plotly, Matplotlib

---

## Disclaimer

This project is for educational purposes only and is not financial advice. Do not use it to make real investment decisions.
