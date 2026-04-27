"""
Day 3 - Collaborative filtering: user-based and item-based neighborhood models.

Builds a user-item rating matrix from MovieLens (with a synthetic fallback),
computes cosine similarity in both directions, and predicts held-out ratings.
We compare the two neighborhood approaches on RMSE/MAE on a small test split.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
ML_PATH = os.path.join(DATA_DIR, "ml-latest-small")
RNG = np.random.default_rng(42)


def load_ratings() -> pd.DataFrame:
    """Load MovieLens ratings if cached locally, otherwise synthesize a small set."""
    if os.path.isdir(ML_PATH):
        ratings = pd.read_csv(os.path.join(ML_PATH, "ratings.csv"))
        return ratings[["userId", "movieId", "rating"]]

    n_users, n_items = 300, 400
    # Latent factors drive synthetic preferences; this gives the matrix real
    # structure that neighborhood methods can actually exploit.
    user_factors = RNG.normal(0, 1, size=(n_users, 4))
    item_factors = RNG.normal(0, 1, size=(n_items, 4))
    full = user_factors @ item_factors.T
    full = 1 + 4 * (full - full.min()) / (full.max() - full.min())

    rows = []
    density = 0.05
    mask = RNG.random((n_users, n_items)) < density
    for u in range(n_users):
        for i in range(n_items):
            if mask[u, i]:
                noise = RNG.normal(0, 0.4)
                rating = float(np.clip(np.round((full[u, i] + noise) * 2) / 2, 0.5, 5.0))
                rows.append((u + 1, i + 1, rating))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def build_matrix(train: pd.DataFrame) -> Tuple[np.ndarray, dict, dict]:
    users = sorted(train["userId"].unique())
    items = sorted(train["movieId"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {m: i for i, m in enumerate(items)}

    matrix = np.zeros((len(users), len(items)), dtype=np.float32)
    for u, m, r in train[["userId", "movieId", "rating"]].itertuples(index=False):
        matrix[u_idx[u], i_idx[m]] = r
    return matrix, u_idx, i_idx


def mean_centered(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract each user's mean rating to remove personal scale bias."""
    mask = matrix > 0
    sums = matrix.sum(axis=1)
    counts = mask.sum(axis=1)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    centered = np.where(mask, matrix - means[:, None], 0.0)
    return centered, means


def predict_user_based(matrix, sims, means, u_i, i_i, k=30) -> float:
    if u_i is None or i_i is None:
        return float(means.mean())
    sim_row = sims[u_i].copy()
    sim_row[u_i] = 0.0  # exclude self

    rated_users = np.where(matrix[:, i_i] > 0)[0]
    if rated_users.size == 0:
        return float(means[u_i])

    s = sim_row[rated_users]
    top_k = np.argsort(-s)[:k]
    neighbors = rated_users[top_k]
    weights = sim_row[neighbors]
    if np.sum(np.abs(weights)) < 1e-8:
        return float(means[u_i])

    deviations = matrix[neighbors, i_i] - means[neighbors]
    pred = means[u_i] + np.dot(weights, deviations) / np.sum(np.abs(weights))
    return float(np.clip(pred, 0.5, 5.0))


def predict_item_based(matrix, sims, u_i, i_i, k=30) -> float:
    if u_i is None or i_i is None:
        return float(matrix[matrix > 0].mean()) if matrix.any() else 3.0
    sim_row = sims[i_i].copy()
    sim_row[i_i] = 0.0

    rated_items = np.where(matrix[u_i] > 0)[0]
    if rated_items.size == 0:
        return float(matrix[matrix > 0].mean())

    s = sim_row[rated_items]
    top_k = np.argsort(-s)[:k]
    neighbors = rated_items[top_k]
    weights = sim_row[neighbors]
    if np.sum(np.abs(weights)) < 1e-8:
        return float(matrix[u_i, rated_items].mean())

    pred = np.dot(weights, matrix[u_i, neighbors]) / np.sum(np.abs(weights))
    return float(np.clip(pred, 0.5, 5.0))


def evaluate(test_df, predict_fn) -> dict:
    truths, preds = [], []
    for u, m, r in test_df[["userId", "movieId", "rating"]].itertuples(index=False):
        preds.append(predict_fn(u, m))
        truths.append(r)
    rmse = float(np.sqrt(mean_squared_error(truths, preds)))
    mae = float(mean_absolute_error(truths, preds))
    return {"rmse": rmse, "mae": mae, "n": len(truths)}


def main():
    print("Loading ratings...")
    ratings = load_ratings()
    print(f"  {len(ratings):,} ratings, {ratings['userId'].nunique()} users, "
          f"{ratings['movieId'].nunique()} movies")

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    matrix, u_idx, i_idx = build_matrix(train)
    print(f"Train matrix: {matrix.shape}, density={np.mean(matrix > 0):.3%}")

    centered, means = mean_centered(matrix)

    print("Computing user-user similarity...")
    user_sims = cosine_similarity(centered)
    print("Computing item-item similarity...")
    item_sims = cosine_similarity(centered.T)

    def user_pred(u, m):
        return predict_user_based(matrix, user_sims, means,
                                  u_idx.get(u), i_idx.get(m))

    def item_pred(u, m):
        return predict_item_based(matrix, item_sims,
                                  u_idx.get(u), i_idx.get(m))

    print("Evaluating user-based CF on test split...")
    user_metrics = evaluate(test, user_pred)
    print(f"  RMSE={user_metrics['rmse']:.4f}  MAE={user_metrics['mae']:.4f}  "
          f"n={user_metrics['n']}")

    print("Evaluating item-based CF on test split...")
    item_metrics = evaluate(test, item_pred)
    print(f"  RMSE={item_metrics['rmse']:.4f}  MAE={item_metrics['mae']:.4f}  "
          f"n={item_metrics['n']}")

    winner = "user-based" if user_metrics["rmse"] < item_metrics["rmse"] else "item-based"
    print(f"Lower RMSE: {winner}")


if __name__ == "__main__":
    main()
