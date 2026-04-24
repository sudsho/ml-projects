"""
Stock Price Forecasting - Day 4: LSTM with PyTorch for Sequence Prediction
Builds a multi-layer LSTM to predict next-day log returns from a lookback window.
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


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")
    else:
        print("Regenerating synthetic stock data…")
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
# 2. Sequence dataset
# ---------------------------------------------------------------------------

LOOKBACK = 20        # input window length
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10      # remaining 10 % is test


def make_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_dataset(series: np.ndarray):
    n = len(series)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return series[:train_end], series[train_end:val_end], series[val_end:]


# ---------------------------------------------------------------------------
# 3. Normalisation (zero-mean, unit-std on train)
# ---------------------------------------------------------------------------

def normalise(train, val, test):
    mu, sigma = train.mean(), train.std() + 1e-8
    return (train - mu) / sigma, (val - mu) / sigma, (test - mu) / sigma, mu, sigma


# ---------------------------------------------------------------------------
# 4. LSTM model
# ---------------------------------------------------------------------------

def build_and_train(X_tr, y_tr, X_val, y_val,
                    hidden=64, num_layers=2, epochs=60, lr=1e-3, batch=32):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch not available — using NumPy AR(20) baseline instead.")
        return _ar_fallback(X_tr, y_tr, X_val, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tensors — shape: (samples, seq_len, 1)
    to_t = lambda a: torch.tensor(a).unsqueeze(-1).to(device)
    tr_ds = TensorDataset(to_t(X_tr), torch.tensor(y_tr).to(device))
    val_ds = TensorDataset(to_t(X_val), torch.tensor(y_val).to(device))
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.2 if num_layers > 1 else 0.0)
            self.fc = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5, verbose=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_loss = []
        for xb, yb in tr_dl:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch_loss.append(loss.item())
        tr_loss = np.mean(batch_loss)

        model.eval()
        with torch.no_grad():
            v_losses = [criterion(model(xb), yb).item() for xb, yb in val_dl]
        v_loss = np.mean(v_losses)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | train={tr_loss:.6f} | val={v_loss:.6f}")

    return model, device, train_losses, val_losses, to_t


def _ar_fallback(X_tr, y_tr, X_val, y_val):
    """Simple linear AR model as a no-PyTorch fallback."""
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
    print("  AR(20) Ridge fallback trained.")
    return reg, None, [], [], None


# ---------------------------------------------------------------------------
# 5. Prediction and evaluation
# ---------------------------------------------------------------------------

def predict(model_out, X_te, device, to_t):
    """Works for both PyTorch model and sklearn fallback."""
    try:
        import torch
        import torch.nn as nn
        model, dev = model_out[0], model_out[1]
        model.eval()
        with torch.no_grad():
            preds = model(to_t(X_te)).cpu().numpy()
        return preds
    except Exception:
        model = model_out[0]
        return model.predict(X_te)


def evaluate(actual, predicted, label, mu=0.0, sigma=1.0):
    # Denormalise
    actual_d = actual * sigma + mu
    pred_d = predicted * sigma + mu
    mae = np.mean(np.abs(actual_d - pred_d))
    rmse = np.sqrt(np.mean((actual_d - pred_d) ** 2))
    print(f"  {label:<30} MAE={mae:.6f}   RMSE={rmse:.6f}")
    return {"label": label, "mae": round(float(mae), 6), "rmse": round(float(rmse), 6)}


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses, out_dir="time-series-stock-prediction"):
    if not train_losses:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses, label="Val loss")
    ax.set_title("LSTM Training Curve", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/lstm_loss_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: lstm_loss_curves.png")


def plot_predictions(actual_norm, pred_norm, dates, mu, sigma,
                     out_dir="time-series-stock-prediction"):
    actual = actual_norm * sigma + mu
    pred = pred_norm * sigma + mu
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, actual, color="black", linewidth=0.9, label="Actual", alpha=0.7)
    ax.plot(dates, pred, color="crimson", linewidth=1.1, label="LSTM forecast")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("LSTM — Test Set Predictions (Log Returns)", fontsize=13)
    ax.set_ylabel("Log Return")
    ax.legend(fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/lstm_predictions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: lstm_predictions.png")


def plot_final_comparison(metrics_list, out_dir="time-series-stock-prediction"):
    labels = [m["label"] for m in metrics_list]
    rmses = [m["rmse"] for m in metrics_list]
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rmses, color=colors[:len(labels)], width=0.5)
    ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=8)
    ax.set_title("All Models — RMSE Comparison", fontsize=13)
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/final_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: final_model_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    series = df["log_returns"].dropna().values

    train_raw, val_raw, test_raw = split_dataset(series)
    train_n, val_n, test_n, mu, sigma = normalise(train_raw, val_raw, test_raw)

    X_tr, y_tr = make_sequences(train_n, LOOKBACK)
    X_val, y_val = make_sequences(val_n, LOOKBACK)
    X_te, y_te = make_sequences(test_n, LOOKBACK)

    print(f"Train seqs: {len(X_tr)} | Val seqs: {len(X_val)} | Test seqs: {len(X_te)}")

    result = build_and_train(X_tr, y_tr, X_val, y_val, epochs=60)

    if len(result) == 5:
        model, device, train_losses, val_losses, to_t = result
        preds_norm = predict((model, device), X_te, device, to_t)
    else:
        model, device, train_losses, val_losses, to_t = result
        preds_norm = model.predict(X_te) if hasattr(model, "predict") else np.zeros_like(y_te)

    lstm_metrics = evaluate(y_te, preds_norm, "LSTM", mu, sigma)

    # Align test dates
    test_dates = df.index[-(len(y_te)):]
    plot_loss_curves(train_losses, val_losses)
    plot_predictions(y_te, preds_norm, test_dates, mu, sigma)

    # Load prior model results for full comparison
    prior_path = Path("time-series-stock-prediction/all_model_results.csv")
    all_metrics = []
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        all_metrics = prior[["label", "mae", "rmse"]].dropna().to_dict("records")
    all_metrics.append(lstm_metrics)
    plot_final_comparison(all_metrics)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv("time-series-stock-prediction/all_model_results.csv", index=False)
    print("\nSaved: all_model_results.csv")
    print("\nDay 4 complete. Next: model comparison, backtesting, and final documentation.")
