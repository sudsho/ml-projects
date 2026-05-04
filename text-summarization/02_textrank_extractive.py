"""
Day 2: Extractive summarization with TextRank.

TextRank treats sentences as nodes in a graph; edges are weighted by
sentence-to-sentence similarity (cosine on TF-IDF vectors here). Running
PageRank over that graph gives us per-sentence importance scores. We pick
the top-k sentences (in original order) as the summary.

Reference: Mihalcea & Tarau, "TextRank: Bringing Order into Texts" (2004).
"""

import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter; good enough for news / wikipedia paragraphs."""
    text = text.strip().replace("\n", " ")
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    # drop sentences that are too short to carry meaning
    return [s for s in sents if len(s.split()) >= 3]


def build_similarity_matrix(sentences: List[str]) -> np.ndarray:
    """Compute pairwise cosine similarity between TF-IDF sentence vectors."""
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    tfidf = vectorizer.fit_transform(sentences)
    sim = cosine_similarity(tfidf)
    # zero out self-similarity so a sentence does not vote for itself
    np.fill_diagonal(sim, 0.0)
    return sim


def power_iteration_pagerank(
    sim: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Stationary distribution of the random surfer with teleport probability."""
    n = sim.shape[0]
    # column-normalize the similarity matrix into a stochastic transition matrix
    col_sums = sim.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    transition = sim / col_sums

    scores = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - damping) / n)

    for _ in range(max_iter):
        new_scores = teleport + damping * (transition @ scores)
        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            scores = new_scores
            break
        scores = new_scores

    return scores


def textrank_summary(text: str, top_k: int = 3) -> Tuple[str, List[Tuple[int, float, str]]]:
    """Return summary string plus the ranked-sentence breakdown for inspection."""
    sentences = split_sentences(text)
    if len(sentences) <= top_k:
        return " ".join(sentences), [(i, 1.0, s) for i, s in enumerate(sentences)]

    sim = build_similarity_matrix(sentences)
    scores = power_iteration_pagerank(sim)

    ranked = sorted(
        zip(range(len(sentences)), scores, sentences),
        key=lambda x: x[1],
        reverse=True,
    )
    top = sorted(ranked[:top_k], key=lambda x: x[0])  # restore original order
    summary = " ".join(s for _, _, s in top)
    return summary, ranked


SAMPLE = (
    "Climate change is one of the most pressing issues of our time. "
    "Rising global temperatures have been linked to extreme weather events. "
    "Scientists agree that human activity is the primary driver. "
    "Carbon dioxide emissions from fossil fuels trap heat in the atmosphere. "
    "Renewable energy sources offer a path toward decarbonization. "
    "Solar and wind power capacity has grown substantially in the last decade. "
    "Many countries have committed to net-zero targets by mid-century. "
    "However, current policies fall short of what is needed. "
    "Adaptation measures will also be necessary as some warming is already locked in. "
    "Public awareness has increased, but action must accelerate."
)


if __name__ == "__main__":
    summary, ranked = textrank_summary(SAMPLE, top_k=3)
    print("=== TextRank summary (top 3) ===")
    print(summary)
    print()
    print("=== Per-sentence scores ===")
    for idx, score, sent in ranked:
        print(f"[{idx:2d}] {score:.4f}  {sent}")
