# Customer Segmentation - Day 3: K-Means with elbow method and silhouette analysis
# Determine optimal number of clusters and fit final K-Means model on scaled PCA features
# Visualize cluster assignments and interpret segment profiles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# -----------------------------------------------------------------
# Reproduce the scaled feature matrix from day 2
# (using the same synthetic mall-customer-style data)
# -----------------------------------------------------------------
def load_and_prepare_data():
    np.random.seed(42)
    n = 400
    age = np.random.randint(18, 70, n)
    income = np.random.normal(55000, 20000, n).clip(15000, 150000)
    spend_score = np.random.randint(1, 101, n)
    purchase_freq = np.random.poisson(8, n).clip(1, 30)
    avg_order_value = (income * 0.002 + spend_score * 3 + np.random.normal(0, 20, n)).clip(20, 600)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income,
        "spending_score": spend_score,
        "purchase_frequency": purchase_freq,
        "avg_order_value": avg_order_value,
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return df, X_scaled, scaler


df, X_scaled, scaler = load_and_prepare_data()

# Reduce to 2D PCA for visualization (same as day 2)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# -----------------------------------------------------------------
# Elbow method: inertia vs k
# -----------------------------------------------------------------
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker="o", linewidth=2, color="steelblue")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-cluster SSE)")
axes[0].set_title("Elbow Method")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), sil_scores, marker="s", linewidth=2, color="tomato")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs k")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("customer-segmentation/elbow_silhouette.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: elbow_silhouette.png")

# Report best k
best_k = list(k_range)[np.argmax(sil_scores)]
print(f"\nBest k by silhouette: {best_k}  (score={max(sil_scores):.4f})")

# -----------------------------------------------------------------
# Fit final K-Means with chosen k
# -----------------------------------------------------------------
K_FINAL = best_k
km_final = KMeans(n_clusters=K_FINAL, n_init=20, random_state=42)
df["cluster"] = km_final.fit_predict(X_scaled)

print(f"\nCluster sizes:\n{df['cluster'].value_counts().sort_index()}")

# -----------------------------------------------------------------
# Silhouette plot for final model
# -----------------------------------------------------------------
sil_vals = silhouette_samples(X_scaled, df["cluster"])
avg_sil = silhouette_score(X_scaled, df["cluster"])

fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, K_FINAL))

for c in range(K_FINAL):
    c_sil = np.sort(sil_vals[df["cluster"] == c])
    size = len(c_sil)
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                     facecolor=colors[c], edgecolor="none", alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size, str(c))
    y_lower = y_upper + 10

ax.axvline(avg_sil, color="red", linestyle="--", label=f"Avg silhouette = {avg_sil:.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette Plot (k={K_FINAL})")
ax.legend()
plt.tight_layout()
plt.savefig("customer-segmentation/silhouette_plot.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: silhouette_plot.png")

# -----------------------------------------------------------------
# Cluster scatter on PCA 2D space
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df["cluster"], cmap="tab10", s=40, alpha=0.7)

# Plot centroids projected to PCA space
centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c="black", marker="X", s=200, zorder=5, label="Centroids")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"K-Means Clusters in PCA Space (k={K_FINAL})")
plt.colorbar(scatter, ax=ax, label="Cluster")
ax.legend()
plt.tight_layout()
plt.savefig("customer-segmentation/kmeans_pca_scatter.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: kmeans_pca_scatter.png")

# -----------------------------------------------------------------
# Cluster profiles: mean of original features per cluster
# -----------------------------------------------------------------
profile_cols = ["age", "annual_income", "spending_score", "purchase_frequency", "avg_order_value"]
profile = df.groupby("cluster")[profile_cols].mean().round(2)
print("\nCluster Profiles (feature means):")
print(profile.to_string())

# Heatmap of normalized profiles
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(profile_norm.T, annot=profile.T, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Normalized value"})
ax.set_title("Cluster Profile Heatmap")
ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig("customer-segmentation/cluster_profile_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: cluster_profile_heatmap.png")

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
print("\n=== K-Means Summary ===")
print(f"Optimal k : {K_FINAL}")
print(f"Inertia   : {km_final.inertia_:.1f}")
print(f"Silhouette: {avg_sil:.4f}")
print("Plots saved to customer-segmentation/")
