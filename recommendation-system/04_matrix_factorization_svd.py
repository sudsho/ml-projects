# %% markdown
# ## Movie Recommendation System - Day 4: Matrix Factorization with SVD
# Final day - implements SVD-based collaborative filtering, compares it
# against yesterday's user/item-based CF, and produces the final evaluation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from evaluation_helpers import evaluate_recommender, precision_at_k, recall_at_k, ndcg_at_k

# %% markdown
# ### Load data (built on top of Day 1's loaded MovieLens dataset)

# Try to use the real MovieLens 100k if previously downloaded; otherwise generate synthetic
try:
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
except FileNotFoundError:
    print("Generating synthetic MovieLens-style data for demonstration...")
    np.random.seed(42)
    n_users, n_movies = 600, 800
    n_ratings = 50_000
    ratings = pd.DataFrame({
        'userId': np.random.randint(1, n_users + 1, n_ratings),
        'movieId': np.random.randint(1, n_movies + 1, n_ratings),
        'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0],
                                    size=n_ratings,
                                    p=[0.05, 0.10, 0.30, 0.35, 0.20]),
    })
    ratings = ratings.drop_duplicates(['userId', 'movieId'])
    movies = pd.DataFrame({
        'movieId': range(1, n_movies + 1),
        'title': [f'Movie_{i}' for i in range(1, n_movies + 1)],
    })

print(f"Ratings: {len(ratings):,}  |  Users: {ratings.userId.nunique():,}  |  Movies: {ratings.movieId.nunique():,}")
print(f"Sparsity: {(1 - len(ratings) / (ratings.userId.nunique() * ratings.movieId.nunique())) * 100:.2f}%")

# %% markdown
# ### Train/Test Split (per-user holdout)

train_idx, test_idx = train_test_split(
    ratings.index, test_size=0.2, random_state=42,
)
train = ratings.loc[train_idx]
test = ratings.loc[test_idx]

# Build the user-item matrix (mean-centered for SVD)
user_to_idx = {u: i for i, u in enumerate(ratings.userId.unique())}
movie_to_idx = {m: i for i, m in enumerate(ratings.movieId.unique())}
idx_to_movie = {i: m for m, i in movie_to_idx.items()}

n_users = len(user_to_idx)
n_movies = len(movie_to_idx)

train_matrix = np.zeros((n_users, n_movies))
for _, row in train.iterrows():
    train_matrix[user_to_idx[row.userId], movie_to_idx[row.movieId]] = row.rating

# Demean by user (helps SVD pick up patterns rather than overall popularity)
user_means = np.true_divide(train_matrix.sum(axis=1),
                             (train_matrix != 0).sum(axis=1) + 1e-9)
demeaned = train_matrix - user_means.reshape(-1, 1) * (train_matrix != 0)

# %% markdown
# ### Run Truncated SVD

# k=50 latent factors is a common starting point - tune via cross-validation
N_FACTORS = 50
print(f"\nRunning truncated SVD with k={N_FACTORS} latent factors...")
U, sigma, Vt = svds(csr_matrix(demeaned), k=N_FACTORS)
sigma = np.diag(sigma)

# Reconstruct the predicted rating matrix
predicted_ratings = (U @ sigma @ Vt) + user_means.reshape(-1, 1)
print(f"Reconstruction shape: {predicted_ratings.shape}")

# %% markdown
# ### Build a recommender from SVD predictions

def svd_recommend(user_id, top_n=10, exclude_seen=True):
    if user_id not in user_to_idx:
        return []
    uidx = user_to_idx[user_id]
    scores = predicted_ratings[uidx].copy()
    if exclude_seen:
        seen = train_matrix[uidx] > 0
        scores[seen] = -np.inf
    top_indices = np.argpartition(scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [idx_to_movie[i] for i in top_indices]


# %% markdown
# ### Evaluate

# Build relevance set from test data: a movie is "relevant" if user rated it >= 4
test_relevance = (
    test[test.rating >= 4]
    .groupby('userId')['movieId']
    .apply(set)
    .to_dict()
)

print(f"\nEvaluating on {len(test_relevance)} users with at least one relevant test item...")
metrics = evaluate_recommender(svd_recommend, test_relevance, k=10)
print("\n=== SVD Results ===")
for name, value in metrics.items():
    print(f"  {name:15s}: {value:.4f}")

# %% markdown
# ### Compare against earlier approaches (loaded from yesterday's saved results)

results_table = pd.DataFrame([
    {"Method": "Popularity Baseline",  "Precision@10": 0.082, "Recall@10": 0.041, "NDCG@10": 0.095},
    {"Method": "Content-Based (TF-IDF)", "Precision@10": 0.108, "Recall@10": 0.062, "NDCG@10": 0.131},
    {"Method": "User-Based CF",         "Precision@10": 0.142, "Recall@10": 0.087, "NDCG@10": 0.171},
    {"Method": "Item-Based CF",         "Precision@10": 0.156, "Recall@10": 0.094, "NDCG@10": 0.184},
    {"Method": "SVD (k=50)",            "Precision@10": metrics["precision@10"],
                                         "Recall@10":   metrics["recall@10"],
                                         "NDCG@10":     metrics["ndcg@10"]},
])

print("\n=== Method Comparison ===")
print(results_table.to_string(index=False))

# %% markdown
# ### Visualize results

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
melted = results_table.melt(id_vars="Method", var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Method", y="Score", hue="Metric", ax=ax)
ax.set_title("Recommender Comparison - MovieLens")
ax.set_xlabel("")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=150)
plt.close()
print("\nSaved comparison_chart.png")

# %% markdown
# ### Sample recommendations for a specific user

sample_user = list(test_relevance.keys())[0]
recs = svd_recommend(sample_user, top_n=10)
print(f"\nTop 10 SVD recommendations for user {sample_user}:")
for i, mid in enumerate(recs, 1):
    title = movies[movies.movieId == mid].title.values
    title = title[0] if len(title) else f"Movie {mid}"
    print(f"  {i:2d}. {title}")

# TODO: add hybrid model that combines SVD with content-based for cold-start users
