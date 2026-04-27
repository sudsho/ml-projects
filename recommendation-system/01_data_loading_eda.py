"""
Day 1 - MovieLens data loading, EDA, and rating distribution analysis.

Loads the MovieLens 100K dataset (or falls back to a synthetic sample if
unavailable), explores rating distributions, user/movie activity, and
identifies sparsity characteristics relevant for recommender design.
"""

import os
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
PLOTS_DIR = "plots"
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def fetch_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    target = os.path.join(DATA_DIR, "ml-latest-small")
    if os.path.isdir(target):
        return target
    try:
        with urllib.request.urlopen(ML_URL, timeout=15) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            z.extractall(DATA_DIR)
        return target
    except Exception as e:
        print(f"[warn] could not fetch MovieLens ({e}); generating synthetic data")
        return None


def load_or_synth(path):
    if path and os.path.isdir(path):
        ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
        movies = pd.read_csv(os.path.join(path, "movies.csv"))
        return ratings, movies

    rng = np.random.default_rng(42)
    n_users, n_movies = 600, 9000
    n_ratings = 80_000
    ratings = pd.DataFrame({
        "userId": rng.integers(1, n_users + 1, n_ratings),
        "movieId": rng.integers(1, n_movies + 1, n_ratings),
        "rating": rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                             size=n_ratings, p=[.02, .03, .04, .07, .08, .15, .18, .22, .12, .09]),
        "timestamp": rng.integers(10**9, 1.7 * 10**9, n_ratings),
    })
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_movies + 1),
        "title": [f"Movie {i}" for i in range(1, n_movies + 1)],
        "genres": rng.choice(["Drama", "Comedy", "Action|Thriller", "Romance",
                              "Sci-Fi|Adventure", "Horror"], n_movies),
    })
    return ratings, movies


def basic_stats(ratings, movies):
    n_users = ratings["userId"].nunique()
    n_movies_rated = ratings["movieId"].nunique()
    sparsity = 1 - (len(ratings) / (n_users * n_movies_rated))
    print("=" * 50)
    print("MovieLens dataset summary")
    print("=" * 50)
    print(f"# ratings        : {len(ratings):,}")
    print(f"# unique users   : {n_users:,}")
    print(f"# unique movies  : {n_movies_rated:,}")
    print(f"matrix sparsity  : {sparsity:.4%}")
    print(f"rating mean/std  : {ratings['rating'].mean():.3f} / {ratings['rating'].std():.3f}")
    print(f"movies in catalog: {len(movies):,}")


def plot_rating_distribution(ratings):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.countplot(x="rating", data=ratings, color="#3b6db0")
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rating_distribution.png"), dpi=120)
    plt.close()


def plot_activity(ratings):
    user_counts = ratings.groupby("userId").size()
    movie_counts = ratings.groupby("movieId").size()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(user_counts, bins=50, color="#3b6db0")
    axes[0].set_yscale("log")
    axes[0].set_title("Ratings per user (log y)")
    axes[0].set_xlabel("Ratings"); axes[0].set_ylabel("Users")

    axes[1].hist(movie_counts, bins=50, color="#b03b6d")
    axes[1].set_yscale("log")
    axes[1].set_title("Ratings per movie (log y)")
    axes[1].set_xlabel("Ratings"); axes[1].set_ylabel("Movies")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "activity_distribution.png"), dpi=120)
    plt.close()


def long_tail_summary(ratings):
    movie_counts = ratings.groupby("movieId").size().sort_values(ascending=False)
    total = movie_counts.sum()
    cum = movie_counts.cumsum() / total
    head_count = (cum < 0.5).sum() + 1
    print(f"\nLong-tail: {head_count} movies (top {head_count/len(movie_counts):.2%}) "
          f"account for 50% of all ratings.")


def main():
    path = fetch_movielens()
    ratings, movies = load_or_synth(path)
    basic_stats(ratings, movies)
    plot_rating_distribution(ratings)
    plot_activity(ratings)
    long_tail_summary(ratings)
    print("\nSaved plots to ./plots/. Day 1 EDA complete.")


if __name__ == "__main__":
    main()
