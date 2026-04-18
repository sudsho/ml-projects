"""
Customer Segmentation - Day 1: Data Loading and Exploratory Data Analysis
Dataset: Synthetic customer data mimicking retail/e-commerce behavior.
Goal: Understand feature distributions before applying clustering algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy import stats

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")

# ─── Generate synthetic customer dataset ─────────────────────────────────────
# Simulates typical RFM-style customer data: Age, Annual Income, Spending Score,
# Purchase Frequency, and Average Order Value.

n_customers = 500

age = np.concatenate([
    np.random.normal(25, 4, 100),   # young adults
    np.random.normal(40, 8, 250),   # mid-age
    np.random.normal(60, 7, 150),   # seniors
]).astype(int)
age = np.clip(age, 18, 80)

income = np.concatenate([
    np.random.normal(30000, 5000, 100),
    np.random.normal(65000, 15000, 250),
    np.random.normal(90000, 20000, 150),
])
income = np.clip(income, 15000, 200000).astype(int)

spending_score = np.concatenate([
    np.random.normal(75, 10, 100),   # young high spenders
    np.random.normal(45, 15, 250),   # average spenders
    np.random.normal(30, 12, 150),   # conservative seniors
])
spending_score = np.clip(spending_score, 1, 100).astype(int)

purchase_freq = np.concatenate([
    np.random.normal(18, 4, 100),
    np.random.normal(10, 3, 250),
    np.random.normal(6, 2, 150),
])
purchase_freq = np.clip(purchase_freq, 1, 30).astype(int)

avg_order_value = (income / 1000 * 0.8 + spending_score * 2 +
                   np.random.normal(0, 20, n_customers))
avg_order_value = np.clip(avg_order_value, 10, 500).round(2)

gender = np.random.choice(["Male", "Female"], size=n_customers, p=[0.48, 0.52])
region = np.random.choice(
    ["North", "South", "East", "West"], size=n_customers, p=[0.3, 0.25, 0.25, 0.2]
)

df = pd.DataFrame({
    "CustomerID": range(1001, 1001 + n_customers),
    "Age": age,
    "Annual_Income": income,
    "Spending_Score": spending_score,
    "Purchase_Frequency": purchase_freq,
    "Avg_Order_Value": avg_order_value,
    "Gender": gender,
    "Region": region,
})

# ─── Basic overview ───────────────────────────────────────────────────────────
print("=" * 60)
print("CUSTOMER SEGMENTATION - EDA")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive statistics:\n{df.describe().round(2)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nGender distribution:\n{df['Gender'].value_counts()}")
print(f"\nRegion distribution:\n{df['Region'].value_counts()}")

# ─── Distribution plots ───────────────────────────────────────────────────────
numeric_cols = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")

for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue", bins=30)
    ax.set_title(col.replace("_", " "))
    ax.set_xlabel("")

# Remove empty subplot
axes[1, 2].set_visible(False)
plt.tight_layout()
plt.savefig("customer-segmentation/feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: feature_distributions.png")

# ─── Correlation heatmap ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("customer-segmentation/correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: correlation_heatmap.png")

# ─── Income vs Spending Score scatter (key clustering view) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df["Annual_Income"], df["Spending_Score"],
                alpha=0.5, c="steelblue", edgecolors="none", s=40)
axes[0].set_xlabel("Annual Income ($)")
axes[0].set_ylabel("Spending Score (1-100)")
axes[0].set_title("Income vs Spending Score")

# Color by gender
colors = {"Male": "steelblue", "Female": "salmon"}
for gender_val, grp in df.groupby("Gender"):
    axes[1].scatter(grp["Annual_Income"], grp["Spending_Score"],
                    label=gender_val, alpha=0.5, s=40, c=colors[gender_val])
axes[1].set_xlabel("Annual Income ($)")
axes[1].set_ylabel("Spending Score (1-100)")
axes[1].set_title("Income vs Spending Score by Gender")
axes[1].legend()

plt.tight_layout()
plt.savefig("customer-segmentation/income_vs_spending.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: income_vs_spending.png")

# ─── Spending score by region ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="Region", y="Spending_Score", data=df, ax=ax, palette="Set2")
ax.set_title("Spending Score Distribution by Region")
plt.tight_layout()
plt.savefig("customer-segmentation/spending_by_region.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: spending_by_region.png")

# ─── Statistical summary for clustering readiness ────────────────────────────
print("\n" + "=" * 60)
print("CLUSTERING READINESS SUMMARY")
print("=" * 60)

for col in numeric_cols:
    skewness = stats.skew(df[col])
    kurtosis = stats.kurtosis(df[col])
    print(f"\n{col}:")
    print(f"  Range:     [{df[col].min()}, {df[col].max()}]")
    print(f"  Skewness:  {skewness:.3f}")
    print(f"  Kurtosis:  {kurtosis:.3f}")

print("\nNote: Features will need scaling before clustering (Day 2 - PCA prep).")
print("\nEDA complete. Dataset saved to memory for downstream steps.")

# Save dataset for subsequent days
df.to_csv("customer-segmentation/customers.csv", index=False)
print("Saved: customers.csv")
