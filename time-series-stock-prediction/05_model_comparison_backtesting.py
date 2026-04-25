"""
Stock Price Forecasting - Day 5: Model Comparison, Backtesting, and Results
Compares ARIMA, SMA, and naive baselines via walk-forward backtesting.
Computes RMSE, MAPE, directional accuracy; runs a simple equity simulation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

TICKER = "MSFT"
TEST_SIZE = 60
OUTPUT_DIR = Path("time-series-stock-prediction/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    data_path = Path("time-series-stock-prediction/processed_stock_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Synthetic MSFT-like price series for reproducibility
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        price = 230.0
        prices = [price]
        for _ in range(999):
            ret = np.random.normal(0.0003, 0.015)
            price = price * np.exp(ret)
            prices.append(round(price, 4))
        df = pd.DataFrame({"Close": prices}, index=dates)
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Forecasting models
# ---------------------------------------------------------------------------

def arima_walk_forward(series: pd.Series, n_test: int = 60, order=(1, 1, 1)) -> np.ndarray:
    """Walk-forward ARIMA: refit on expanding window each step."""
    history = list(series.iloc[:-n_test])
    preds = []
    for t in range(n_test):
        model = ARIMA(history, order=order)
        fit = model.fit()
        yhat = float(fit.forecast(steps=1)[0])
        preds.append(yhat)
        history.append(float(series.iloc[len(series) - n_test + t]))
    return np.array(preds)


def naive_forecast(series: pd.Series, n_test: int = 60) -> np.ndarray:
    """Random-walk baseline: next = last observed."""
    start = len(series) - n_test - 1
    return np.array([float(series.iloc[start + t]) for t in range(n_test)])


def sma_forecast(series: pd.Series, n_test: int = 60, window: int = 5) -> np.ndarray:
    """Rolling mean over `window` days as forecast."""
    preds = []
    for t in range(n_test):
        idx = len(series) - n_test + t
        preds.append(float(series.iloc[idx - window:idx].mean()))
    return np.array(preds)


# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------

def metrics(actual: np.ndarray, pred: np.ndarray, name: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(np.mean(np.abs(actual - pred)))
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
    dir_acc = float(np.mean((np.diff(actual) > 0) == (np.diff(pred) > 0)) * 100)
    return {"Model": name, "RMSE": round(rmse, 3), "MAE": round(mae, 3),
            "MAPE(%)": round(mape, 2), "DirAcc(%)": round(dir_acc, 2)}


# ---------------------------------------------------------------------------
# 4. Simple long/cash equity backtest
# ---------------------------------------------------------------------------

def backtest(prices: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Go long when prediction exceeds last close, hold cash otherwise."""
    equity = 1.0
    curve = []
    for i in range(1, len(prices)):
        signal = 1 if preds[i - 1] > prices[i - 1] else 0
        daily_ret = (prices[i] - prices[i - 1]) / prices[i - 1]
        equity *= 1 + signal * daily_ret
        curve.append(equity)
    return np.array(curve)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_forecasts(actual: pd.Series, forecasts: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(range(len(actual)), actual.values, label="Actual", color="black", lw=2)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, preds), color in zip(forecasts.items(), colors):
        ax.plot(range(len(preds)), preds, label=name, ls="--", color=color, alpha=0.85)
    ax.set_title(f"{TICKER} — Walk-Forward Forecast Comparison (last {TEST_SIZE} days)")
    ax.set_xlabel("Trading day (test window)")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_equity_curves(curves: dict, buy_hold: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(buy_hold, label="Buy & Hold", color="black", lw=2)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, curve), color in zip(curves.items(), colors):
        ax.plot(curve, label=name, ls="--", color=color, alpha=0.85)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_title("Equity Curves (normalized to 1.0)")
    ax.set_xlabel("Trading day (test window)")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_metrics(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, col in zip(axes, ["RMSE", "MAPE(%)", "DirAcc(%)"]):
        bars = ax.bar(df["Model"], df[col], color=palette)
        ax.set_title(col, fontsize=12)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, str(val),
                    ha="center", va="bottom", fontsize=9)
    plt.suptitle(f"{TICKER} Model Metrics — {TEST_SIZE}-Day Test Window", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    close = df["Close"]
    actual = close.iloc[-TEST_SIZE:]
    actual_arr = actual.values

    print(f"\n{'='*60}")
    print(f"  {TICKER} Stock Forecasting — Final Comparison (Day 5)")
    print(f"  Training: {len(close) - TEST_SIZE} days | Test: {TEST_SIZE} days")
    print(f"{'='*60}\n")

    print("Fitting ARIMA(1,1,1) walk-forward (this may take ~30s)...")
    arima_preds = arima_walk_forward(close, n_test=TEST_SIZE)

    naive_preds = naive_forecast(close, n_test=TEST_SIZE)
    sma_preds = sma_forecast(close, n_test=TEST_SIZE)

    # --- Metrics table ---
    results = [
        metrics(actual_arr, arima_preds, "ARIMA(1,1,1)"),
        metrics(actual_arr, naive_preds, "Naive"),
        metrics(actual_arr, "SMA(5)"),
    ]

    # fix: pass correct arg for SMA
    results = [
        metrics(actual_arr, arima_preds, "ARIMA(1,1,1)"),
        metrics(actual_arr, naive_preds, "Naive"),
        metrics(actual_arr, sma_preds, "SMA(5)"),
    ]
    metrics_df = pd.DataFrame(results)
    print("--- Forecast Metrics ---")
    print(metrics_df.to_string(index=False))

    # --- Backtesting ---
    equity_arima = backtest(actual_arr, arima_preds)
    equity_naive = backtest(actual_arr, naive_preds)
    equity_sma = backtest(actual_arr, sma_preds)
    buy_hold_curve = actual_arr[1:] / actual_arr[0]

    print(f"\n--- Backtest Final Equity (start = 1.00) ---")
    print(f"  ARIMA strategy : {equity_arima[-1]:.4f}")
    print(f"  Naive strategy : {equity_naive[-1]:.4f}")
    print(f"  SMA(5) strategy: {equity_sma[-1]:.4f}")
    print(f"  Buy & Hold     : {buy_hold_curve[-1]:.4f}")

    # --- Save outputs ---
    metrics_df.to_csv(OUTPUT_DIR / "final_metrics.csv", index=False)

    forecasts_dict = {
        "ARIMA(1,1,1)": arima_preds,
        "Naive": naive_preds,
        "SMA(5)": sma_preds,
    }
    plot_forecasts(actual, forecasts_dict, OUTPUT_DIR / "forecast_comparison.png")
    plot_equity_curves(
        {"ARIMA": equity_arima, "Naive": equity_naive, "SMA(5)": equity_sma},
        buy_hold_curve, OUTPUT_DIR / "equity_curves.png"
    )
    plot_metrics(metrics_df, OUTPUT_DIR / "metrics_bar.png")

    print(f"\nAll outputs written to {OUTPUT_DIR}/")
    print("Stock forecasting project complete.\n")
