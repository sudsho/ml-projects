"""
Day 2: Feature Engineering - Encoding, Scaling, and Feature Selection
House Price Prediction Project

Builds on the EDA from Day 1. Constructs new features from the California
Housing dataset, handles outliers, applies scaling, and selects the most
informative features using correlation, variance, and mutual information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load dataset (same as Day 1)
# ─────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df.columns = [
    "median_income", "house_age", "avg_rooms", "avg_bedrooms",
    "population", "avg_occupancy", "latitude", "longitude", "median_house_value"
]

print("=== Day 2: Feature Engineering ===")
print(f"Raw shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. Outlier capping (winsorization at 1%/99%)
# ─────────────────────────────────────────────
outlier_cols = ["avg_rooms", "avg_bedrooms", "population", "avg_occupancy"]
for col in outlier_cols:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

print("\nOutlier capping applied to:", outlier_cols)

# ─────────────────────────────────────────────
# 3. Construct new features
# ─────────────────────────────────────────────
# Rooms per person — density metric
df["rooms_per_person"] = df["avg_rooms"] / (df["avg_occupancy"] + 1e-6)

# Bedroom ratio — higher ratio = smaller, less desirable units
df["bedroom_ratio"] = df["avg_bedrooms"] / (df["avg_rooms"] + 1e-6)

# Population density proxy
df["pop_per_room"] = df["population"] / (df["avg_rooms"] + 1e-6)

# Income × location interaction (income matters more in high-latitude areas)
df["income_lat"] = df["median_income"] * df["latitude"]

# Log income — reduces right skew, often improves linear model fit
df["log_income"] = np.log1p(df["median_income"])

# Distance from San Francisco (lat 37.77, lon -122.42) — coastal premium
df["dist_sf"] = np.sqrt(
    (df["latitude"] - 37.77) ** 2 + (df["longitude"] + 122.42) ** 2
)

# Distance from Los Angeles (lat 34.05, lon -118.24)
df["dist_la"] = np.sqrt(
    (df["latitude"] - 34.05) ** 2 + (df["longitude"] + 118.24) ** 2
)

# Age bucket — older homes may have distinct pricing patterns
df["old_home"] = (df["house_age"] >= 40).astype(int)

new_features = [
    "rooms_per_person", "bedroom_ratio", "pop_per_room",
    "income_lat", "log_income", "dist_sf", "dist_la", "old_home"
]
print(f"\nNew features created ({len(new_features)}): {new_features}")

# ─────────────────────────────────────────────
# 4. Log-transform the target (right-skewed)
# ─────────────────────────────────────────────
df["log_price"] = np.log1p(df["median_house_value"])
print(f"\nTarget skewness (raw):  {df['median_house_value'].skew():.3f}")
print(f"Target skewness (log):  {df['log_price'].skew():.3f}")

# ─────────────────────────────────────────────
# 5. Train / validation split (before scaling)
# ─────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ["median_house_value", "log_price"]]
X = df[feature_cols]
y = df["log_price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# ─────────────────────────────────────────────
# 6. Scaling — RobustScaler (handles remaining outliers well)
# ─────────────────────────────────────────────
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val), columns=feature_cols, index=X_val.index
)
print("\nFeatures scaled with RobustScaler (fit on train only)")

# ─────────────────────────────────────────────
# 7. Feature selection — mutual information
# ─────────────────────────────────────────────
mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_df = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

print("\n=== Mutual Information Scores (vs log_price) ===")
for feat, score in mi_df.items():
    bar = "█" * int(score * 30)
    print(f"  {feat:<22} {score:.4f}  {bar}")

# Keep top features (MI score > 0.05 or top 12)
top_features = mi_df[mi_df > 0.05].index.tolist()
if len(top_features) < 6:
    top_features = mi_df.head(8).index.tolist()

print(f"\nSelected {len(top_features)} features: {top_features}")

# ─────────────────────────────────────────────
# 8. Correlation of new features with target
# ─────────────────────────────────────────────
corr_all = df[feature_cols + ["log_price"]].corr()["log_price"].drop("log_price").sort_values(key=abs, ascending=False)

print("\n=== Pearson Correlation with log(price) ===")
for feat, val in corr_all.items():
    bar = "█" * int(abs(val) * 25)
    tag = " ← NEW" if feat in new_features else ""
    print(f"  {feat:<22} {val:+.4f}  {bar}{tag}")

# ─────────────────────────────────────────────
# 9. Visualize: MI scores bar chart
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mi_df.plot(kind="bar", ax=axes[0], color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("Mutual Information Scores", fontsize=12)
axes[0].set_ylabel("MI Score")
axes[0].tick_params(axis="x", rotation=45)
axes[0].axhline(0.05, color="red", linestyle="--", linewidth=1, label="threshold=0.05")
axes[0].legend()

corr_all.plot(kind="bar", ax=axes[1], color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].set_title("Pearson Correlation with log(Price)", fontsize=12)
axes[1].set_ylabel("Correlation")
axes[1].tick_params(axis="x", rotation=45)
axes[1].axhline(0, color="black", linewidth=0.8)

plt.suptitle("Feature Importance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: feature_importance.png")

# ─────────────────────────────────────────────
# 10. Save processed data for Day 3
# ─────────────────────────────────────────────
X_train_final = X_train_scaled[top_features]
X_val_final = X_val_scaled[top_features]

train_out = X_train_final.copy()
train_out["log_price"] = y_train
val_out = X_val_final.copy()
val_out["log_price"] = y_val

train_out.to_csv("train_features.csv", index=False)
val_out.to_csv("val_features.csv", index=False)

print(f"\nSaved: train_features.csv ({train_out.shape})")
print(f"Saved: val_features.csv  ({val_out.shape})")
print(f"\n=== Day 2 Complete: Feature Engineering done ===")
print(f"  • {len(new_features)} new features constructed")
print(f"  • Outlier capping + RobustScaler applied")
print(f"  • {len(top_features)} top features selected via mutual information")
print(f"  • Log-transformed target (skew: {df['median_house_value'].skew():.2f} → {df['log_price'].skew():.2f})")
