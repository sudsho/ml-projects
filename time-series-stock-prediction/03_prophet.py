"""
Stock Price Forecasting - Day 3: Facebook Prophet Implementation and Comparison
Fits a Prophet model on price levels, tunes seasonality, and compares vs ARIMA baseline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

DATA_PATH = Path("time-series-stock-prediction/processed_stock_data.csv")
ARIMA_RESULTS = Path("time-series-stock-prediction/arima_results.csv")


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")
    else:
        print("processed_stock_data.csv not found — regenerating synthetic data.")
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
# 2. Prepare Prophet dataframe (requires columns ds, y)
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.85


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    prophet_df = df[["price"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    return prophet_df


def train_test_split_prophet(prophet_df: pd.DataFrame):
    split = int(len(prophet_df) * TRAIN_FRAC)
    return prophet_df.iloc[:split].copy(), prophet_df.iloc[split:].copy()


# ---------------------------------------------------------------------------
# 3. Fit Prophet model
# ---------------------------------------------------------------------------

def fit_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet
        except ImportError:
            print("Prophet not installed — using linear trend simulation as fallback.")
            return _prophet_fallback(train_df, test_df)

    print(f"\nFitting Prophet on {len(train_df)} training points…")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,   # regularise trend changepoints
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    model.fit(train_df)

    # Forecast over the test horizon
    future = model.make_future_dataframe(periods=len(test_df), freq="B")
    forecast = model.predict(future)
    forecast_test = forecast.iloc[-len(test_df):].reset_index(drop=True)
    return model, forecast, forecast_test


def _prophet_fallback(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Linear extrapolation stand-in when Prophet is not installed."""
    from sklearn.linear_model import LinearRegression
    X_tr = np.arange(len(train_df)).reshape(-1, 1)
    lr = LinearRegression().fit(X_tr, train_df["y"].values)
    X_te = np.arange(len(train_df), len(train_df) + len(test_df)).reshape(-1, 1)
    preds = lr.predict(X_te)
    forecast_test = test_df.copy().reset_index(drop=True)
    forecast_test["yhat"] = preds
    forecast_test["yhat_lower"] = preds * 0.95
    forecast_test["yhat_upper"] = preds * 1.05
    print("  (fallback) linear trend used instead of Prophet")
    return None, None, forecast_test


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate(actual: np.ndarray, predicted: np.ndarray, label: str) -> dict:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    print(f"  {label:<30} MAE={mae:.4f}   RMSE={rmse:.4f}   MAPE={mape:.2f}%")
    return {"label": label, "mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_prophet_forecast(train_df, test_df, forecast_test,
                           out_dir="time-series-stock-prediction"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_df["ds"], train_df["y"], color="black", linewidth=0.8,
            label="Train (actual)", alpha=0.6)
    ax.plot(test_df["ds"].values, test_df["y"].values, color="steelblue",
            linewidth=1, label="Test (actual)")
    ax.plot(test_df["ds"].values, forecast_test["yhat"].values,
            color="darkorange", linewidth=1.2, label="Prophet forecast")
    ax.fill_between(test_df["ds"].values,
                    forecast_test["yhat_lower"].values,
                    forecast_test["yhat_upper"].values,
                    alpha=0.2, color="darkorange", label="95% CI")
    ax.set_title("Prophet — Out-of-Sample Price Forecast", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prophet_forecast.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: prophet_forecast.png")


def plot_components(model, forecast, out_dir="time-series-stock-prediction"):
    if model is None:
        print("Skipping component plot (Prophet not available).")
        return
    fig = model.plot_components(forecast)
    fig.suptitle("Prophet Components — Trend, Weekly, Yearly", y=1.01, fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/prophet_components.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: prophet_components.png")


def plot_model_comparison(metrics_list: list, out_dir="time-series-stock-prediction"):
    labels = [m["label"] for m in metrics_list]
    maes = [m["mae"] for m in metrics_list]
    rmses = [m["rmse"] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Model Comparison — MAE and RMSE", fontsize=13)

    for ax, vals, title in zip(axes, [maes, rmses], ["MAE", "RMSE"]):
        bars = ax.bar(x, vals, width=0.5, color=["steelblue", "darkorange", "seagreen"][:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")


# ---------------------------------------------------------------------------
# 6. Changepoint analysis
# ---------------------------------------------------------------------------

def print_changepoints(model, n: int = 5):
    if model is None:
        return
    try:
        deltas = model.params["delta"].mean(axis=0)
        cp_df = pd.DataFrame({
            "changepoint": model.changepoints,
            "magnitude": deltas,
        }).sort_values("magnitude", key=abs, ascending=False)
        print(f"\nTop {n} trend changepoints:")
        print(cp_df.head(n).to_string(index=False))
    except Exception as e:
        print(f"Could not extract changepoints: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    prophet_df = prepare_prophet_df(df)
    train_df, test_df = train_test_split_prophet(prophet_df)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    result = fit_prophet(train_df, test_df)
    if len(result) == 3:
        model, forecast, forecast_test = result
    else:
        model, forecast, forecast_test = None, None, result

    # Evaluate
    print("\n--- Evaluation ---")
    prophet_metrics = evaluate(
        test_df["y"].values,
        forecast_test["yhat"].values,
        "Prophet"
    )

    # Load ARIMA metrics if available for comparison
    all_metrics = [prophet_metrics]
    if ARIMA_RESULTS.exists():
        arima_df = pd.read_csv(ARIMA_RESULTS)
        for _, row in arima_df.iterrows():
            all_metrics.insert(0, {
                "label": row["label"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": None,
            })
        print("\nLoaded ARIMA/SARIMA metrics for comparison:")
        for m in all_metrics:
            mape_str = f"  MAPE={m['mape']:.2f}%" if m.get("mape") else ""
            print(f"  {m['label']:<30} MAE={m['mae']:.4f}   RMSE={m['rmse']:.4f}{mape_str}")

    # Plots
    print("\nGenerating plots…")
    plot_prophet_forecast(train_df, test_df, forecast_test)
    plot_components(model, forecast)
    plot_model_comparison([m for m in all_metrics if m.get("mape") is not None or m.get("rmse")])
    print_changepoints(model)

    # Save combined results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv("time-series-stock-prediction/all_model_results.csv", index=False)
    print("\nSaved: all_model_results.csv")
    print("\nDay 3 complete. Next: LSTM approach with PyTorch for sequence prediction.")
