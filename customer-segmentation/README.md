# Customer Segmentation with Clustering

Unsupervised learning project applying K-Means, DBSCAN, and PCA to segment customers into actionable behavioral groups.

## Project Overview

| | |
|---|---|
| **Type** | Unsupervised Learning / Clustering |
| **Dataset** | Synthetic customer data (200 customers, 5 features) |
| **Techniques** | K-Means, DBSCAN, PCA, Silhouette Analysis |
| **Libraries** | scikit-learn, pandas, matplotlib, seaborn |

## Daily Breakdown

| Day | Script | Description |
|-----|--------|-------------|
| 1 | `01_eda.py` | Data loading, distribution plots, correlation heatmap, outlier detection |
| 2 | `02_feature_scaling_pca.py` | StandardScaler, PCA variance explained, 2D/3D projections |
| 3 | `03_kmeans_clustering.py` | Elbow method, silhouette scores, optimal k selection, initial cluster viz |
| 4 | `04_dbscan_profiling.py` | DBSCAN parameter grid search, K-Means vs DBSCAN comparison, cluster profiling |

## Key Results

- **Optimal clusters**: K=4 via silhouette analysis
- **K-Means outperformed DBSCAN** on this dataset — customer features form roughly spherical clusters with similar densities, which favors K-Means
- **DBSCAN** effectively identified noise/outlier customers (~5–10% of data)

## Customer Segments Identified

| Segment | Income | Spending | Profile |
|---------|--------|----------|---------|
| High Value Loyalists | High | High | Core revenue drivers, high CLV |
| Budget Enthusiasts | Low | High | Frequent buyers, price-sensitive |
| Affluent but Cautious | High | Low | Potential upsell targets |
| Low Engagement | Low | Low | Churn risk, needs reactivation |

## Visualizations

- `clustering_final_analysis.png` — 6-panel final analysis: PCA scatter, cluster sizes, feature heatmap, income vs spending segmentation

## Skills Demonstrated

- Unsupervised learning pipeline: scaling → dimensionality reduction → clustering
- Hyperparameter tuning for DBSCAN (epsilon, min_samples grid search)
- Cluster evaluation: Silhouette score, Davies-Bouldin index
- Business interpretation of statistical segments
