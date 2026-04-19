"""
Day 2 - Customer Segmentation: Feature Scaling and Dimensionality Reduction with PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


# ── reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

# ── load / recreate the customer dataset from Day 1 ────────────────────────────
def load_customer_data():
    """Recreate the synthetic customer dataset used in EDA."""
    np.random.seed(42)
    n = 500

    data = {
        "customer_id": range(1, n + 1),
        "age": np.random.normal(40, 12, n).clip(18, 80).astype(int),
        "annual_income": np.random.normal(60000, 20000, n).clip(15000, 150000),
        "spending_score": np.random.randint(1, 101, n),
        "num_purchases": np.random.poisson(15, n),
        "avg_order_value": np.random.exponential(80, n).clip(10, 500),
        "days_since_last_purchase": np.random.randint(1, 365, n),
        "loyalty_years": np.random.uniform(0, 10, n).round(1),
    }

    df = pd.DataFrame(data)
    # introduce mild correlations
    df.loc[df["spending_score"] > 70, "annual_income"] *= np.random.uniform(1.1, 1.4, (df["spending_score"] > 70).sum())
    df.loc[df["loyalty_years"] > 5, "num_purchases"] += np.random.randint(5, 15, (df["loyalty_years"] > 5).sum())
    return df


df = load_customer_data()
features = ["age", "annual_income", "spending_score", "num_purchases",
            "avg_order_value", "days_since_last_purchase", "loyalty_years"]
X = df[features].copy()

print("Dataset shape:", X.shape)
print("\nFeature statistics before scaling:")
print(X.describe().round(2))

# ── 1. Feature Scaling ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

# StandardScaler: zero mean, unit variance — assumes roughly normal distribution
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# RobustScaler: uses median/IQR — better for skewed features like avg_order_value
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
X_robust_df = pd.DataFrame(X_robust, columns=features)

print("\nAfter StandardScaler - means (should be ~0):")
print(X_scaled_df.mean().round(4).to_string())
print("\nAfter StandardScaler - stds (should be ~1):")
print(X_scaled_df.std().round(4).to_string())

# visualize distributions before vs. after scaling
fig, axes = plt.subplots(2, len(features), figsize=(20, 6))
for i, feat in enumerate(features):
    axes[0, i].hist(X[feat], bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0, i].set_title(feat, fontsize=8)
    axes[0, i].set_xlabel("Original", fontsize=7)

    axes[1, i].hist(X_scaled_df[feat], bins=30, color="coral", edgecolor="white", linewidth=0.5)
    axes[1, i].set_xlabel("Scaled", fontsize=7)

axes[0, 0].set_ylabel("Count")
axes[1, 0].set_ylabel("Count")
plt.suptitle("Feature Distributions: Before vs After StandardScaling", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("02_scaling_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: 02_scaling_distributions.png")

# ── 2. Correlation heatmap on scaled features ──────────────────────────────────
corr = X_scaled_df.corr()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Matrix (Scaled)", fontsize=13)
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 02_correlation_heatmap.png")

# ── 3. PCA ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)

pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("\nExplained variance per component:")
for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
    bar = "█" * int(ev * 50)
    print(f"  PC{i+1}: {ev:.4f} ({cv:.4f} cumulative)  {bar}")

# find how many components for 90% / 95% variance
n_90 = np.argmax(cumulative_var >= 0.90) + 1
n_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nComponents for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")

# scree + cumulative variance plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.bar(range(1, len(explained_var) + 1), explained_var, color="steelblue", edgecolor="white")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance Ratio")
ax1.set_title("Scree Plot")
ax1.set_xticks(range(1, len(explained_var) + 1))

ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, "o-", color="coral", linewidth=2)
ax2.axhline(0.90, ls="--", color="gray", alpha=0.7, label="90% threshold")
ax2.axhline(0.95, ls="--", color="green", alpha=0.7, label="95% threshold")
ax2.axvline(n_90, ls=":", color="gray", alpha=0.5)
ax2.axvline(n_95, ls=":", color="green", alpha=0.5)
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.set_title("Cumulative Variance Explained")
ax2.legend()
ax2.set_xticks(range(1, len(cumulative_var) + 1))

plt.tight_layout()
plt.savefig("02_pca_variance.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 02_pca_variance.png")

# ── 4. PCA biplot (PC1 vs PC2) ─────────────────────────────────────────────────
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"\nPC1 explained variance: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"PC2 explained variance: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"Total (2D):             {pca_2d.explained_variance_ratio_.sum():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# scatter of projections
axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.4, s=20, color="steelblue")
axes[0].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
axes[0].set_title("2D PCA Projection of Customer Data")
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].axvline(0, color="gray", lw=0.5)

# loading plot (biplot arrows)
loadings = pca_2d.components_.T
scale = 3.0
for j, feat in enumerate(features):
    axes[1].arrow(0, 0, loadings[j, 0] * scale, loadings[j, 1] * scale,
                  head_width=0.08, head_length=0.05, fc="coral", ec="coral")
    axes[1].text(loadings[j, 0] * scale * 1.12, loadings[j, 1] * scale * 1.12,
                 feat, fontsize=9, ha="center")
axes[1].set_xlim(-3.5, 3.5)
axes[1].set_ylim(-3.5, 3.5)
axes[1].set_xlabel("PC1 loadings")
axes[1].set_ylabel("PC2 loadings")
axes[1].set_title("PCA Loading Plot (Feature Contributions)")
axes[1].axhline(0, color="gray", lw=0.5)
axes[1].axvline(0, color="gray", lw=0.5)

plt.tight_layout()
plt.savefig("02_pca_biplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 02_pca_biplot.png")

# ── 5. 3-component PCA for clustering context ──────────────────────────────────
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"\nVariance captured with 3 PCs: {pca_3d.explained_variance_ratio_.sum():.4f}")

# component interpretation
print("\nTop feature loadings per principal component:")
loading_df = pd.DataFrame(
    pca_full.components_[:3],
    columns=features,
    index=[f"PC{i+1}" for i in range(3)]
)
print(loading_df.round(3).to_string())

# save scaled data and PCA projections for Day 3 (clustering)
np.save("X_scaled.npy", X_scaled)
np.save("X_pca_2d.npy", X_pca_2d)
np.save("X_pca_3d.npy", X_pca_3d)
print("\nSaved preprocessed arrays: X_scaled.npy, X_pca_2d.npy, X_pca_3d.npy")

print("\nDay 2 complete. Ready for clustering in Day 3.")
