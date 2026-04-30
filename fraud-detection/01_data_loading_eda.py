"""
Credit Card Fraud Detection - Day 1
Data loading, severe class imbalance analysis and visualization.

Dataset: Kaggle credit card fraud dataset (creditcard.csv) with PCA-anonymized
features V1-V28, plus 'Time', 'Amount', and 'Class' (target: 0=legit, 1=fraud).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification


DATA_PATH = "data/creditcard.csv"
PLOTS_DIR = "plots"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the credit card dataset, falling back to a synthetic imbalanced sample."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded real dataset: {df.shape}")
        return df

    print("Real dataset not found - generating synthetic imbalanced sample.")
    X, y = make_classification(
        n_samples=50000,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        weights=[0.998, 0.002],
        flip_y=0.001,
        random_state=42,
    )
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = np.abs(df["Amount"]) * 100
    df["Time"] = np.arange(len(df))
    df["Class"] = y
    return df


def basic_summary(df: pd.DataFrame) -> None:
    print("\n=== Shape and dtypes ===")
    print(df.shape)
    print(df.dtypes.value_counts())

    print("\n=== Missing values ===")
    print(df.isna().sum().sum(), "total NaNs")

    print("\n=== Class distribution ===")
    counts = df["Class"].value_counts()
    print(counts)
    fraud_pct = counts.get(1, 0) / len(df) * 100
    print(f"Fraud rate: {fraud_pct:.4f}%")
    print(f"Imbalance ratio: 1 fraud per {len(df) // max(counts.get(1, 1), 1)} legit")


def plot_class_imbalance(df: pd.DataFrame, out_dir: str = PLOTS_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x="Class", data=df, ax=axes[0])
    axes[0].set_title("Class counts (linear)")
    axes[0].set_yscale("linear")

    sns.countplot(x="Class", data=df, ax=axes[1])
    axes[1].set_title("Class counts (log scale)")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_imbalance.png"), dpi=120)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, out_dir: str = PLOTS_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    axes[0].hist(legit, bins=50, alpha=0.6, label="Legit", color="steelblue")
    axes[0].hist(fraud, bins=50, alpha=0.8, label="Fraud", color="crimson")
    axes[0].set_yscale("log")
    axes[0].set_title("Transaction amount by class")
    axes[0].set_xlabel("Amount")
    axes[0].legend()

    sns.boxplot(x="Class", y="Amount", data=df, ax=axes[1])
    axes[1].set_title("Amount boxplot by class")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "amount_by_class.png"), dpi=120)
    plt.close(fig)


def feature_correlation_summary(df: pd.DataFrame) -> pd.Series:
    """Top features correlated with the fraud label."""
    corr = df.corr(numeric_only=True)["Class"].drop("Class")
    top = corr.abs().sort_values(ascending=False).head(10)
    print("\n=== Top 10 features correlated with Class ===")
    print(top)
    return top


def main() -> None:
    df = load_data()
    basic_summary(df)
    plot_class_imbalance(df)
    plot_amount_distribution(df)
    feature_correlation_summary(df)
    print("\nDay 1 EDA complete. Plots saved to ./plots/")


if __name__ == "__main__":
    main()
