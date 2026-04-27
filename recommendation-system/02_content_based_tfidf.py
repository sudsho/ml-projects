"""
Day 2 - Content-based filtering using TF-IDF on movie metadata.

Builds a content profile for each movie from its genres (and title tokens),
computes a TF-IDF matrix, and recommends similar movies using cosine
similarity. Falls back to a synthetic movie catalog when MovieLens isn't
available locally.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
ML_PATH = os.path.join(DATA_DIR, "ml-latest-small")


def load_movies():
    if os.path.isdir(ML_PATH):
        movies = pd.read_csv(os.path.join(ML_PATH, "movies.csv"))
    else:
        rng = np.random.default_rng(7)
        n = 2000
        genres_pool = ["Drama", "Comedy", "Action", "Thriller", "Romance",
                       "Sci-Fi", "Adventure", "Horror", "Animation", "Crime"]
        movies = pd.DataFrame({
            "movieId": np.arange(1, n + 1),
            "title": [f"Synthetic Movie {i} ({1980 + i % 40})" for i in range(1, n + 1)],
            "genres": ["|".join(rng.choice(genres_pool, size=rng.integers(1, 4), replace=False))
                       for _ in range(n)],
        })
    return movies


def build_content_field(movies: pd.DataFrame) -> pd.Series:
    # combine genre tokens and lightly cleaned title tokens
    def clean_title(t: str) -> str:
        t = re.sub(r"\(\d{4}\)", "", str(t))   # drop year
        t = re.sub(r"[^a-zA-Z0-9 ]", " ", t)
        return t.lower().strip()

    genre_tokens = movies["genres"].fillna("").str.replace("|", " ", regex=False).str.lower()
    title_tokens = movies["title"].apply(clean_title)
    # weight genres more by repeating them
    return genre_tokens + " " + genre_tokens + " " + title_tokens


def build_tfidf(corpus: pd.Series):
    vec = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    print(f"TF-IDF matrix: {matrix.shape[0]} docs x {matrix.shape[1]} terms "
          f"(nnz={matrix.nnz:,}, density={matrix.nnz / np.prod(matrix.shape):.4%})")
    return vec, matrix


def recommend(title_query: str, movies: pd.DataFrame, matrix, top_k: int = 10):
    titles = movies["title"].astype(str)
    mask = titles.str.contains(title_query, case=False, regex=False)
    if not mask.any():
        print(f"No movie matched '{title_query}'.")
        return pd.DataFrame()

    idx = int(np.argmax(mask.values))
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    sims[idx] = -1.0   # exclude the query movie itself
    top = np.argsort(-sims)[:top_k]

    out = movies.iloc[top][["movieId", "title", "genres"]].copy()
    out["similarity"] = sims[top].round(4)
    print(f"\nQuery: {titles.iloc[idx]}\nGenres: {movies.iloc[idx]['genres']}")
    print(out.to_string(index=False))
    return out


def main():
    movies = load_movies()
    print(f"Loaded {len(movies):,} movies.")

    corpus = build_content_field(movies)
    _, matrix = build_tfidf(corpus)

    # try a few representative queries
    for q in ["Toy Story", "Matrix", "Godfather", "Synthetic Movie 1"]:
        recommend(q, movies, matrix, top_k=8)

    print("\nDay 2 content-based recommender complete.")


if __name__ == "__main__":
    main()
