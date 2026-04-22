"""
Stock Price Forecasting - Day 2: ARIMA/SARIMA Modeling and Diagnostics
Fits ARIMA and SARIMA models on log-returns, evaluates residuals and forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

DATA_PATH = Path("time-series-stock-prediction/processed_stock_data.csv")


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")
    else:
        # Regenerate synthetic data if Day 1 output is missing
        print("Processed data not found — regenerating synthetic series.")
        np.random.seed(42)
        n = 1260
        dates = pd.bdate_range(end=pd.Timestamp("2026-04-21"), periods=n)
        returns = np.random.normal(0.0003, 0.015, n)
        price = 150 * np.cumprod(1 + returns)
        df = pd.DataFrame({"price": price}, index=dates)
        df.index.name = "date"
        df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
        df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# 2. Auto ARIMA order selection via AIC grid search
# ---------------------------------------------------------------------------

def select_arima_order(series: pd.Series, max_p: int = 4, max_q: int = 4) -> tuple:
    best_aic = np.inf
    best_order = (1, 0, 1)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                res = ARIMA(series, order=(p, 0, q)).fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, 0, q)
            except Exception:
                continue
    print(f"Best ARIMA order by AIC: {best_order}  (AIC={best_aic:.2f})")
    return best_order


# ---------------------------------------------------------------------------
# 3. Train / test split helpers
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.85


def train_test_split_ts(series: pd.Series):
    split = int(len(series) * TRAIN_FRAC)
    return series.iloc[:split], series.iloc[split:]


def evaluate(actual: pd.Series, predicted: np.ndarray, label: str):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    print(f"  {label:<25} MAE={mae:.6f}   RMSE={rmse:.6f}")
    return {"label": label, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# 4. ARIMA fit and rolling forecast
# ---------------------------------------------------------------------------

def fit_arima(train: pd.Series, test: pd.Series, order: tuple) -> dict:
    print(f"\nFitting ARIMA{order} on {len(train)} training points…")
    model = ARIMA(train, order=order)
    result = model.fit()
    print(result.summary().tables[0])

    # Rolling one-step-ahead forecast
    history = list(train)
    preds = []
    for obs in test:
        m = ARIMA(history, order=order).fit()
        preds.append(m.forecast(steps=1)[0])
        history.append(obs)

    preds = np.array(preds)
    metrics = evaluate(test, preds, f"ARIMA{order}")
    return {"result": result, "preds": preds, "metrics": metrics}


# ---------------------------------------------------------------------------
# 5. SARIMA fit (weekly seasonality m=5 trading days)
# ---------------------------------------------------------------------------

def fit_sarima(train: pd.Series, test: pd.Series) -> dict:
    order = (1, 0, 1)
    seasonal_order = (1, 0, 1, 5)
    print(f"\nFitting SARIMA{order}x{seasonal_order} …")
    result = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    # In-sample + out-of-sample forecast
    forecast_obj = result.get_forecast(steps=len(test))
    preds = forecast_obj.predicted_mean.values
    ci = forecast_obj.conf_int()

    metrics = evaluate(test, preds, f"SARIMA{order}x{seasonal_order}")
    return {"result": result, "preds": preds, "ci": ci, "metrics": metrics}


# ---------------------------------------------------------------------------
# 6. Residual diagnostics
# ---------------------------------------------------------------------------

def plot_diagnostics(arima_result, sarima_result, out_dir: str = "time-series-stock-prediction"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ARIMA Residual Diagnostics", fontsize=13, fontweight="bold")

    residuals = arima_result.resid
    axes[0, 0].plot(residuals.index, residuals, linewidth=0.8)
    axes[0, 0].axhline(0, color="red", linewidth=0.8)
    axes[0, 0].set_title("Residuals over time")

    axes[0, 1].hist(residuals, bins=50, edgecolor="white")
    axes[0, 1].set_title("Residual distribution")

    plot_acf(residuals, lags=30, ax=axes[1, 0], title="ACF of Residuals")
    plot_pacf(residuals, lags=30, ax=axes[1, 1], title="PACF of Residuals")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/arima_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: arima_diagnostics.png")

    # Ljung-Box test
    lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print("\nLjung-Box test on ARIMA residuals:")
    print(lb.to_string())


def plot_forecast_comparison(test: pd.Series, arima_preds, sarima_preds, sarima_ci,
                              out_dir: str = "time-series-stock-prediction"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("ARIMA vs SARIMA — Out-of-Sample Forecast (Log Returns)", fontsize=13)

    for ax, preds, label, color in zip(
        axes,
        [arima_preds, sarima_preds],
        ["ARIMA", "SARIMA"],
        ["steelblue", "darkorange"],
    ):
        ax.plot(test.index, test.values, color="black", linewidth=0.8, label="Actual", alpha=0.7)
        ax.plot(test.index, preds, color=color, linewidth=1.2, label=f"{label} forecast")
        if label == "SARIMA" and sarima_ci is not None:
            ax.fill_between(test.index,
                            sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1],
                            alpha=0.2, color=color, label="95% CI")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8)
        ax.set_ylabel("Log Return")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/arima_vs_sarima_forecast.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: arima_vs_sarima_forecast.png")


# ---------------------------------------------------------------------------
# 7. ADF re-check on the series being modeled
# ---------------------------------------------------------------------------

def check_stationarity(series: pd.Series, name: str):
    stat, p, *_ = adfuller(series.dropna())
    status = "STATIONARY" if p < 0.05 else "NON-STATIONARY"
    print(f"ADF on {name}: stat={stat:.4f}  p={p:.4f}  => {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    series = df["log_returns"].dropna()

    check_stationarity(series, "log_returns")

    train, test = train_test_split_ts(series)
    print(f"Train: {len(train)} | Test: {len(test)}")

    # Grid-search for best ARIMA order (light search on train subset for speed)
    best_order = select_arima_order(train.iloc[-200:], max_p=3, max_q=3)

    arima_out = fit_arima(train, test, best_order)
    sarima_out = fit_sarima(train, test)

    print("\n--- Model Comparison ---")
    for m in [arima_out["metrics"], sarima_out["metrics"]]:
        print(f"  {m['label']:<30} MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}")

    plot_diagnostics(arima_out["result"], sarima_out["result"])
    plot_forecast_comparison(test, arima_out["preds"], sarima_out["preds"],
                              sarima_out.get("ci"))

    # Save results summary
    results_df = pd.DataFrame([arima_out["metrics"], sarima_out["metrics"]])
    results_df.to_csv("time-series-stock-prediction/arima_results.csv", index=False)
    print("\nSaved: arima_results.csv")
    print("\nDay 2 complete. Next: Facebook Prophet implementation and comparison.")
