# %% markdown
# ## Recommendation System - Helper utilities for evaluation
# Reusable functions for computing precision@k, recall@k, NDCG@k.
# Will be used in tomorrow's matrix factorization comparison.

import numpy as np
from typing import List, Set


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Fraction of relevant items captured in top-k."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain - weights position."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Binary - did at least one relevant item appear in top-k?"""
    return float(any(item in relevant for item in recommended[:k]))


def mean_reciprocal_rank(recommended: List[int], relevant: Set[int]) -> float:
    """Reciprocal of the rank of the first relevant item."""
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_recommender(predict_fn, test_data, k=10):
    """
    Run all metrics on a test set.

    predict_fn: callable(user_id) -> List[item_id] sorted by score desc
    test_data: dict { user_id: set(relevant_item_ids) }
    """
    metrics = {"precision": [], "recall": [], "ndcg": [], "hit_rate": [], "mrr": []}

    for user_id, relevant in test_data.items():
        if not relevant:
            continue
        recs = predict_fn(user_id)
        metrics["precision"].append(precision_at_k(recs, relevant, k))
        metrics["recall"].append(recall_at_k(recs, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recs, relevant, k))
        metrics["hit_rate"].append(hit_rate(recs, relevant, k))
        metrics["mrr"].append(mean_reciprocal_rank(recs))

    return {f"{name}@{k}" if name != "mrr" else "mrr": np.mean(values)
            for name, values in metrics.items()}


# Quick sanity tests
if __name__ == "__main__":
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 7}

    print(f"Precision@5: {precision_at_k(recommended, relevant, 5):.3f}")  # 0.4
    print(f"Recall@5:    {recall_at_k(recommended, relevant, 5):.3f}")     # 0.667
    print(f"NDCG@5:      {ndcg_at_k(recommended, relevant, 5):.3f}")
    print(f"Hit Rate@5:  {hit_rate(recommended, relevant, 5):.3f}")        # 1.0
    print(f"MRR:         {mean_reciprocal_rank(recommended, relevant):.3f}")  # 0.5
