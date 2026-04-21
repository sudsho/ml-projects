"""
Stock Price Forecasting - Day 1: Data Fetching, EDA, and Stationarity Tests
Fetches historical stock data using yfinance and performs comprehensive time series EDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# 1. Data loading (uses yfinance if available, else generates synthetic data)
# ---------------------------------------------------------------------------

def load_stock_data(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df = df[["Close"]].rename(columns={"Close": "price"})
        print(f"Downloaded {len(df)} trading days of {ticker} data via yfinance.")
    except Exception:
        # Fallback: simulate realistic stock prices with geometric Brownian motion
        print("yfinance unavailable — generating synthetic stock data.")
        np.random.seed(42)
        n = 1260  # ~5 years of trading days
        dates = pd.bdate_range(end=pd.Timestamp("2026-04-21"), periods=n)
        returns = np.random.normal(0.0003, 0.015, n)
        price = 150 * np.cumprod(1 + returns)
        df = pd.DataFrame({"price": price}, index=dates)
        ticker = "AAPL (simulated)"

    df.index.name = "date"
    return df, ticker


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
    df["rolling_mean_20"] = df["price"].rolling(20).mean()
    df["rolling_std_20"] = df["price"].rolling(20).std()
    df["rolling_mean_50"] = df["price"].rolling(50).mean()
    df["upper_bb"] = df["rolling_mean_20"] + 2 * df["rolling_std_20"]
    df["lower_bb"] = df["rolling_mean_20"] - 2 * df["rolling_std_20"]
    return df.dropna()


# ---------------------------------------------------------------------------
# 2. Stationarity tests
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, name: str = "series") -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    output = {
        "series": name,
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "lags": result[2],
        "n_obs": result[3],
        "stationary_5pct": result[1] < 0.05,
    }
    for key, val in result[4].items():
        output[f"critical_{key}"] = round(val, 4)
    return output


def kpss_test(series: pd.Series, name: str = "series") -> dict:
    stat, p_val, lags, crits = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "series": name,
        "kpss_stat": round(stat, 4),
        "p_value": round(p_val, 4),
        "lags": lags,
        # KPSS null = stationary; reject null (p < 0.05) → non-stationary
        "stationary_5pct": p_val >= 0.05,
    }


def print_stationarity_report(df: pd.DataFrame):
    series_to_test = {
        "price": df["price"],
        "log_returns": df["log_returns"],
        "returns": df["returns"],
    }
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS")
    print("=" * 60)
    for name, series in series_to_test.items():
        adf = adf_test(series, name)
        kp = kpss_test(series, name)
        print(f"\n[{name.upper()}]")
        print(f"  ADF  stat={adf['adf_stat']:>8.4f}  p={adf['p_value']:.4f}  "
              f"=> {'STATIONARY' if adf['stationary_5pct'] else 'NON-STATIONARY'}")
        print(f"  KPSS stat={kp['kpss_stat']:>8.4f}  p={kp['p_value']:.4f}  "
              f"=> {'STATIONARY' if kp['stationary_5pct'] else 'NON-STATIONARY'}")


# ---------------------------------------------------------------------------
# 3. Visualizations
# ---------------------------------------------------------------------------

def plot_price_overview(df: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{ticker} — Price Overview (5 Years)", fontsize=14, fontweight="bold")

    # Price with Bollinger Bands
    axes[0].plot(df.index, df["price"], label="Close", linewidth=1)
    axes[0].plot(df.index, df["rolling_mean_20"], label="MA20", linewidth=1, linestyle="--")
    axes[0].plot(df.index, df["rolling_mean_50"], label="MA50", linewidth=1, linestyle=":")
    axes[0].fill_between(df.index, df["lower_bb"], df["upper_bb"], alpha=0.15, label="Bollinger Bands")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(fontsize=8)

    # Daily returns
    axes[1].bar(df.index, df["returns"], width=1, alpha=0.6, color="steelblue")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Daily Returns")

    # Rolling volatility (20-day std of log returns)
    rolling_vol = df["log_returns"].rolling(20).std() * np.sqrt(252)
    axes[2].plot(df.index, rolling_vol, color="darkorange", linewidth=1)
    axes[2].set_ylabel("Annualized Vol")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("time-series-stock-prediction/price_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: price_overview.png")


def plot_decomposition(df: pd.DataFrame, ticker: str):
    # Decompose on weekly-resampled prices to reduce noise
    weekly = df["price"].resample("W").last()
    decomp = seasonal_decompose(weekly, model="multiplicative", period=52)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{ticker} — Seasonal Decomposition (Weekly)", fontsize=13, fontweight="bold")
    for ax, comp, label in zip(
        axes,
        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        ax.plot(comp.index, comp, linewidth=1)
        ax.set_ylabel(label)
    plt.tight_layout()
    plt.savefig("time-series-stock-prediction/decomposition.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: decomposition.png")


def plot_acf_pacf(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ACF / PACF — Price vs Log Returns", fontsize=13, fontweight="bold")

    plot_acf(df["price"], lags=50, ax=axes[0, 0], title="ACF — Price")
    plot_pacf(df["price"], lags=50, ax=axes[0, 1], title="PACF — Price")
    plot_acf(df["log_returns"], lags=50, ax=axes[1, 0], title="ACF — Log Returns")
    plot_pacf(df["log_returns"], lags=50, ax=axes[1, 1], title="PACF — Log Returns")

    plt.tight_layout()
    plt.savefig("time-series-stock-prediction/acf_pacf.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: acf_pacf.png")


def plot_returns_distribution(df: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{ticker} — Returns Distribution", fontsize=13, fontweight="bold")

    sns.histplot(df["log_returns"], bins=80, kde=True, ax=axes[0])
    axes[0].set_title("Log Returns Histogram")
    axes[0].set_xlabel("Log Return")

    from scipy import stats
    stats.probplot(df["log_returns"].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot vs Normal")

    plt.tight_layout()
    plt.savefig("time-series-stock-prediction/returns_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: returns_distribution.png")


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, ticker: str):
    print("\n" + "=" * 60)
    print(f"DATASET SUMMARY — {ticker}")
    print("=" * 60)
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Observations: {len(df)}")
    print(f"\nPrice stats:")
    print(f"  Min    : ${df['price'].min():.2f}")
    print(f"  Max    : ${df['price'].max():.2f}")
    print(f"  Mean   : ${df['price'].mean():.2f}")
    print(f"  Std    : ${df['price'].std():.2f}")
    print(f"\nReturn stats (log returns):")
    lr = df["log_returns"]
    print(f"  Mean   : {lr.mean():.5f}  ({lr.mean()*252:.2%} annualized)")
    print(f"  Std    : {lr.std():.5f}  ({lr.std()*np.sqrt(252):.2%} annualized vol)")
    print(f"  Skew   : {lr.skew():.4f}")
    print(f"  Kurtosis: {lr.kurtosis():.4f}")
    worst = lr.nsmallest(5)
    best = lr.nlargest(5)
    print(f"\n  Worst 5 days: {worst.values.round(4).tolist()}")
    print(f"  Best  5 days: {best.values.round(4).tolist()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_raw, ticker = load_stock_data("AAPL", period="5y")
    df = compute_features(df_raw)

    print_summary(df, ticker)
    print_stationarity_report(df)

    print("\nGenerating plots...")
    plot_price_overview(df, ticker)
    plot_decomposition(df, ticker)
    plot_acf_pacf(df)
    plot_returns_distribution(df, ticker)

    # Save processed data for next day's modeling
    df.to_csv("time-series-stock-prediction/processed_stock_data.csv")
    print("\nSaved: processed_stock_data.csv")
    print("\nDay 1 complete. Next: ARIMA/SARIMA modeling and diagnostics.")
