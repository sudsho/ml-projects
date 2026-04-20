import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# --- Recreate dataset (same seed as previous days) ---
n_customers = 200
data = pd.DataFrame({
    'Age': np.random.randint(18, 70, n_customers),
    'Annual_Income': np.random.randint(15, 137, n_customers),
    'Spending_Score': np.random.randint(1, 100, n_customers),
    'Purchase_Frequency': np.random.randint(1, 50, n_customers),
    'Avg_Basket_Size': np.random.randint(20, 500, n_customers)
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# --- DBSCAN parameter grid search ---
print("=== DBSCAN Parameter Search ===")
eps_values = [0.4, 0.6, 0.8, 1.0, 1.2]
min_samples_values = [3, 5, 8]

dbscan_results = []
for eps in eps_values:
    for min_s in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_s)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        if n_clusters >= 2:
            sil = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
            dbscan_results.append({
                'eps': eps, 'min_samples': min_s,
                'n_clusters': n_clusters, 'n_noise': n_noise,
                'silhouette': round(sil, 4), 'davies_bouldin': round(dbi, 4)
            })

if dbscan_results:
    dbscan_df = pd.DataFrame(dbscan_results).sort_values('silhouette', ascending=False)
    print(dbscan_df.head(8).to_string(index=False))
    best_row = dbscan_df.iloc[0]
    best_eps = best_row['eps']
    best_min_s = int(best_row['min_samples'])
else:
    print("No valid DBSCAN configurations found, using defaults")
    best_eps, best_min_s = 0.8, 5

best_db = DBSCAN(eps=best_eps, min_samples=best_min_s)
dbscan_labels = best_db.fit_predict(X_scaled)
n_db_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"\nBest DBSCAN config: eps={best_eps}, min_samples={best_min_s}")
print(f"  Clusters found: {n_db_clusters}, Noise points: {n_noise} ({n_noise/n_customers*100:.1f}%)")

# --- K-Means reference model (k=4 from day 3) ---
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# --- Metrics comparison ---
print("\n=== Algorithm Comparison ===")
km_sil = silhouette_score(X_scaled, kmeans_labels)
km_dbi = davies_bouldin_score(X_scaled, kmeans_labels)
print(f"K-Means (k={best_k}):  Silhouette={km_sil:.4f}  Davies-Bouldin={km_dbi:.4f}")

if n_db_clusters >= 2:
    db_sil = silhouette_score(X_scaled, dbscan_labels)
    db_dbi = davies_bouldin_score(X_scaled, dbscan_labels)
    print(f"DBSCAN:          Silhouette={db_sil:.4f}  Davies-Bouldin={db_dbi:.4f}")
    # silhouette: higher = better; davies-bouldin: lower = better
    winner = "K-Means" if km_sil > db_sil else "DBSCAN"
    print(f"  -> {winner} achieves better cluster separation on this dataset")

# --- Cluster profiling ---
data_km = data.copy()
data_km['Cluster'] = kmeans_labels

print("\n=== K-Means Cluster Profiles ===")
profile = data_km.groupby('Cluster').agg(
    Count=('Age', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Income=('Annual_Income', 'mean'),
    Avg_Spending=('Spending_Score', 'mean'),
    Avg_Frequency=('Purchase_Frequency', 'mean'),
    Avg_Basket=('Avg_Basket_Size', 'mean')
).round(1)
print(profile.to_string())

# Assign descriptive segment names based on income + spending quadrants
median_income = data['Annual_Income'].median()
median_spending = data['Spending_Score'].median()

segment_names = {}
for cid in range(best_k):
    row = profile.loc[cid]
    high_inc = row['Avg_Income'] >= median_income
    high_spend = row['Avg_Spending'] >= median_spending
    if high_inc and high_spend:
        segment_names[cid] = 'High Value Loyalists'
    elif not high_inc and high_spend:
        segment_names[cid] = 'Budget Enthusiasts'
    elif high_inc and not high_spend:
        segment_names[cid] = 'Affluent but Cautious'
    else:
        segment_names[cid] = 'Low Engagement'

print("\nSegment labels:")
for cid, name in segment_names.items():
    print(f"  Cluster {cid}: {name} (n={int(profile.loc[cid, 'Count'])})")

# --- Visualizations ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
colors = plt.cm.tab10(np.linspace(0, 0.5, best_k))

# 1. K-Means in PCA space
sc1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.65, s=45)
centers_pca = pca.transform(kmeans.cluster_centers_)
axes[0, 0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=150, zorder=5)
axes[0, 0].set_title(f'K-Means (k={best_k}) — PCA Projection')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
plt.colorbar(sc1, ax=axes[0, 0], label='Cluster')

# 2. DBSCAN in PCA space
sc2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='tab10', alpha=0.65, s=45)
axes[0, 1].set_title(f'DBSCAN (eps={best_eps}, min_s={best_min_s}) — PCA Projection\n(-1 = noise)')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
plt.colorbar(sc2, ax=axes[0, 1], label='Cluster')

# 3. Cluster size distribution
cluster_sizes = data_km['Cluster'].value_counts().sort_index()
bars = axes[0, 2].bar(cluster_sizes.index, cluster_sizes.values, color=[plt.cm.tab10(i / 10) for i in range(best_k)], edgecolor='black', alpha=0.85)
axes[0, 2].set_title('Customer Count per Cluster')
axes[0, 2].set_xlabel('Cluster')
axes[0, 2].set_ylabel('Count')
for bar, val in zip(bars, cluster_sizes.values):
    axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha='center', va='bottom', fontsize=10)

# 4. Heatmap of normalized cluster profiles
profile_features = profile[['Avg_Age', 'Avg_Income', 'Avg_Spending', 'Avg_Frequency', 'Avg_Basket']]
profile_norm = (profile_features - profile_features.min()) / (profile_features.max() - profile_features.min())
sns.heatmap(profile_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 0],
            linewidths=0.5, cbar_kws={'label': 'Normalized Value'})
axes[1, 0].set_title('Normalized Cluster Feature Heatmap')
axes[1, 0].set_yticklabels([f"C{i}: {segment_names[i][:12]}" for i in range(best_k)], rotation=0)

# 5. Income vs Spending scatter (most interpretable view)
for cid in range(best_k):
    mask = kmeans_labels == cid
    axes[1, 1].scatter(data['Annual_Income'][mask], data['Spending_Score'][mask],
                       alpha=0.65, s=45, label=f"C{cid}: {segment_names[cid]}")
axes[1, 1].set_title('Annual Income vs Spending Score by Segment')
axes[1, 1].set_xlabel('Annual Income (k$)')
axes[1, 1].set_ylabel('Spending Score')
axes[1, 1].legend(fontsize=7, loc='upper left')
axes[1, 1].axhline(median_spending, color='gray', linestyle='--', alpha=0.4)
axes[1, 1].axvline(median_income, color='gray', linestyle='--', alpha=0.4)

# 6. Grouped bar — feature comparison across clusters
features_to_plot = ['Avg_Age', 'Avg_Income', 'Avg_Spending', 'Avg_Frequency']
x = np.arange(len(features_to_plot))
width = 0.18
for i in range(best_k):
    vals = profile_norm.iloc[i][['Avg_Age', 'Avg_Income', 'Avg_Spending', 'Avg_Frequency']].values
    axes[1, 2].bar(x + i * width, vals, width, label=f'C{i}', alpha=0.85)
axes[1, 2].set_xticks(x + width * (best_k - 1) / 2)
axes[1, 2].set_xticklabels(['Age', 'Income', 'Spending', 'Frequency'], fontsize=10)
axes[1, 2].set_title('Normalized Feature Comparison by Cluster')
axes[1, 2].set_ylabel('Normalized Value')
axes[1, 2].legend(fontsize=8)

plt.suptitle('Customer Segmentation — Final Analysis (Day 4)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('customer-segmentation/clustering_final_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: customer-segmentation/clustering_final_analysis.png")

print("\n=== PROJECT COMPLETE ===")
print(f"Customers: {n_customers} | Features: {data.shape[1]} | Final segments: {best_k}")
print(f"K-Means Silhouette: {km_sil:.4f} (higher is better)")
print(f"K-Means Davies-Bouldin: {km_dbi:.4f} (lower is better)")
print("Segments: " + ", ".join(segment_names.values()))
