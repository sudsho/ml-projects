# Stock Price Forecasting

Time series analysis and forecasting for stock prices using classical statistical models and deep learning.

## Project Overview

This project explores multiple approaches to stock price prediction — from classical ARIMA/SARIMA models through Facebook Prophet to an LSTM neural network with PyTorch. The final day benchmarks all approaches with walk-forward backtesting.

## Daily Breakdown

| Day | Script | Description |
|-----|--------|-------------|
| 1 | `01_data_fetching_eda.py` | Data loading, time series EDA, stationarity tests (ADF, KPSS) |
| 2 | `02_arima_sarima.py` | ARIMA/SARIMA modeling, ACF/PACF analysis, residual diagnostics |
| 3 | `03_prophet.py` | Facebook Prophet with trend + seasonality decomposition |
| 4 | `04_lstm.py` | Multi-layer LSTM in PyTorch for sequence prediction |
| 5 | `05_model_comparison_backtesting.py` | Walk-forward backtest, metrics comparison, equity curves |

## Models Compared

- **ARIMA(1,1,1)** — classical differenced autoregressive model
- **Facebook Prophet** — additive decomposition with holiday effects
- **LSTM** — sequence-to-one deep learning on log returns
- **Naive / SMA(5)** — baselines for sanity-checking

## Key Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error on price levels |
| MAPE | Mean absolute percentage error |
| Directional Accuracy | % of days where price direction was predicted correctly |
| Equity curve | Long/cash strategy vs buy-and-hold |

## Tech Stack

- `pandas`, `numpy` — data manipulation
- `statsmodels` — ARIMA/SARIMA, stationarity tests
- `prophet` — trend + seasonality decomposition
- `torch` — LSTM model
- `scikit-learn` — metrics, preprocessing
- `matplotlib` — visualization
