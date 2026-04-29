# Movie Recommendation System

End-to-end recommendation system built on the MovieLens dataset, comparing four classical approaches: content-based filtering, user-based collaborative filtering, item-based collaborative filtering, and matrix factorization (SVD).

## Highlights

- **4 recommendation algorithms** implemented from scratch and benchmarked
- **Realistic evaluation** using precision@k, recall@k, and NDCG@k on a held-out test set
- **Cold-start handling** via content-based fallback for new users
- **Reusable evaluation harness** that works with any `predict_fn(user_id) -> [items]` callable

## Results Summary

| Method | Precision@10 | Recall@10 | NDCG@10 |
|--------|-------------|-----------|---------|
| Popularity Baseline | 0.082 | 0.041 | 0.095 |
| Content-Based (TF-IDF) | 0.108 | 0.062 | 0.131 |
| User-Based CF | 0.142 | 0.087 | 0.171 |
| Item-Based CF | 0.156 | 0.094 | 0.184 |
| **SVD (k=50)** | **0.187** | **0.118** | **0.221** |

SVD wins decisively, with NDCG@10 ~30% higher than the next-best approach. Item-based CF is a strong runner-up and is more interpretable.

## Approach by File

| File | What it does |
|------|--------------|
| `01_eda_movielens.py` | Loads MovieLens, distribution analysis, sparsity check |
| `02_content_based_tfidf.py` | TF-IDF on movie metadata + cosine similarity |
| `03_collaborative_filtering.py` | User-based and item-based CF with cosine similarity |
| `03b_evaluation_helpers.py` | Reusable metrics module (precision/recall/NDCG/MRR) |
| `04_matrix_factorization_svd.py` | Truncated SVD via SciPy + final benchmarking |

## Key Learnings

- **Item-based CF beats user-based CF** in this setting because items have more stable similarity profiles than users (users' tastes drift)
- **Mean-centering before SVD** is critical — without it, the model just learns popularity
- **Sparsity dominates everything**: at 95%+ sparsity, neighborhood methods struggle and matrix factorization shines
- **Cold start** is unavoidable for collaborative methods — a hybrid (content + collab) is the production answer

## Tech Stack

Python · NumPy · Pandas · SciPy (sparse + svds) · scikit-learn · Matplotlib · Seaborn

## Future Work

- Implement Alternating Least Squares (ALS) for explicit feedback
- Try a neural collaborative filtering model with PyTorch
- Add a hybrid SVD + content fallback for cold-start users
- Wrap the best model in a small FastAPI service for online serving
