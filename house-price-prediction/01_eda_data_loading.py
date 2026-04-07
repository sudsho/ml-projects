"""
Day 1: Dataset Loading, EDA with Correlation Analysis and Visualizations
House Price Prediction Project

Loads the California Housing dataset, performs thorough EDA including
distribution analysis, correlation heatmap, geographic scatter plot,
and identifies key features driving house prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df.columns = [
    "median_income", "house_age", "avg_rooms", "avg_bedrooms",
    "population", "avg_occupancy", "latitude", "longitude", "median_house_value"
]

print("=== California Housing Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nFeatures: {list(df.columns)}")
print(f"\nTarget: median_house_value (in $100k units)")

# ─────────────────────────────────────────────
# 2. Basic statistics
# ─────────────────────────────────────────────
print("\n=== Descriptive Statistics ===")
print(df.describe().round(3).to_string())

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget range: ${df['median_house_value'].min()*100:.0f}k - ${df['median_house_value'].max()*100:.0f}k")
print(f"Target mean:  ${df['median_house_value'].mean()*100:.0f}k")
print(f"Target median:${df['median_house_value'].median()*100:.0f}k")

# ─────────────────────────────────────────────
# 3. Target distribution
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["median_house_value"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("House Value Distribution", fontsize=12)
axes[0].set_xlabel("Median House Value ($100k)")
axes[0].set_ylabel("Count")
axes[0].axvline(df["median_house_value"].mean(), color="red", linestyle="--", label=f"Mean={df['median_house_value'].mean():.2f}")
axes[0].legend()

# Log-transformed target
axes[1].hist(np.log1p(df["median_house_value"]), bins=50, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].set_title("Log-Transformed House Value", fontsize=12)
axes[1].set_xlabel("log(1 + Median House Value)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("target_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: target_distribution.png")

# ─────────────────────────────────────────────
# 4. Feature distributions
# ─────────────────────────────────────────────
features = ["median_income", "house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(features):
    axes[i].hist(df[feat].clip(upper=df[feat].quantile(0.99)), bins=40,
                 color="#55A868", edgecolor="white", alpha=0.8)
    axes[i].set_title(feat.replace("_", " ").title(), fontsize=10)
    axes[i].set_xlabel("Value")

plt.suptitle("Feature Distributions (clipped at 99th percentile)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: feature_distributions.png")

# ─────────────────────────────────────────────
# 5. Correlation heatmap
# ─────────────────────────────────────────────
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax
)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: correlation_heatmap.png")

# ─────────────────────────────────────────────
# 6. Top correlations with target
# ─────────────────────────────────────────────
target_corr = corr["median_house_value"].drop("median_house_value").sort_values(key=abs, ascending=False)
print("\n=== Feature Correlations with House Value ===")
for feat, val in target_corr.items():
    bar = "█" * int(abs(val) * 20)
    print(f"  {feat:<20} {val:+.4f}  {bar}")

# ─────────────────────────────────────────────
# 7. Geographic price distribution
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    df["longitude"], df["latitude"],
    c=df["median_house_value"], cmap="YlOrRd",
    s=df["population"] / 500, alpha=0.4, linewidths=0
)
plt.colorbar(scatter, ax=ax, label="Median House Value ($100k)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("California Housing Prices by Location\n(dot size = population)", fontsize=12)
plt.tight_layout()
plt.savefig("geographic_price_map.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: geographic_price_map.png")

# ─────────────────────────────────────────────
# 8. Income vs House Value scatter
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["median_income"], df["median_house_value"], alpha=0.15, s=5, color="#4C72B0")
ax.set_xlabel("Median Income")
ax.set_ylabel("Median House Value ($100k)")
ax.set_title("Income vs House Value (strongest predictor)", fontsize=11)
plt.tight_layout()
plt.savefig("income_vs_price.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: income_vs_price.png")

# ─────────────────────────────────────────────
# 9. Save cleaned dataset
# ─────────────────────────────────────────────
df.to_csv("housing_data.csv", index=False)
print(f"\nSaved dataset: housing_data.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
print("\n=== Day 1 Complete: EDA finished ===")
print("Key findings:")
print("  • median_income is the strongest predictor (r=+0.69)")
print("  • Coastal areas (low longitude, mid latitude) have highest prices")
print("  • Target is right-skewed; log transform may improve regression")
print("  • No missing values — clean dataset ready for feature engineering")
