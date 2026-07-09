"""
Day 1 of Word2Vec (Skip-Gram with Negative Sampling) from scratch.

Builds the text side of the project before any model exists: tokenize a corpus,
count word frequencies, drop rare words below a min-count, subsample very
frequent words the way Mikolov et al. (2013) do, and generate (center, context)
training pairs with a dynamic window. We keep everything in NumPy / plain Python
so the mechanics stay visible.
"""

import re
from collections import Counter

import numpy as np


# A tiny built-in corpus so the script runs with no downloads. Swap in a real
# text file via load_corpus() when training for real.
SAMPLE_TEXT = """
the quick brown fox jumps over the lazy dog while the dog sleeps in the sun
a king rules the kingdom and a queen rules beside the king in the palace
paris is the capital of france and berlin is the capital of germany today
machine learning models learn word vectors from large amounts of raw text
""".strip()


def tokenize(text):
    """Lowercase and split on runs of non-letter characters."""
    return [tok for tok in re.split(r"[^a-z]+", text.lower()) if tok]


def load_corpus(path=None):
    if path is None:
        return tokenize(SAMPLE_TEXT)
    with open(path, "r", encoding="utf-8") as f:
        return tokenize(f.read())


class Vocabulary:
    """Word <-> id mapping with frequency-based filtering."""

    def __init__(self, tokens, min_count=1):
        counts = Counter(tokens)
        # Keep only words seen at least min_count times, most frequent first so
        # low ids correspond to common words (handy for negative sampling).
        kept = [(w, c) for w, c in counts.most_common() if c >= min_count]
        self.itos = [w for w, _ in kept]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([c for _, c in kept], dtype=np.float64)
        self.total = float(self.counts.sum())

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens if t in self.stoi]


def subsample_keep_prob(counts, total, threshold=1e-3):
    """Mikolov subsampling: probability of KEEPING each word.

    Frequent words (high f = count/total) are discarded more aggressively. The
    published formula is P(keep) = sqrt(t / f) + t / f, clipped to [0, 1].
    """
    freq = counts / total
    with np.errstate(divide="ignore"):
        ratio = threshold / freq
    keep = np.sqrt(ratio) + ratio
    return np.clip(keep, 0.0, 1.0)


def subsample(ids, keep_prob, rng):
    """Drop tokens according to their per-word keep probability."""
    draws = rng.random(len(ids))
    return [wid for wid, u in zip(ids, draws) if u < keep_prob[wid]]


def generate_pairs(ids, max_window, rng):
    """Yield (center, context) id pairs with a dynamically shrunk window.

    For each position a window size in 1..max_window is drawn uniformly, which
    up-weights nearby words - the standard skip-gram trick.
    """
    pairs = []
    n = len(ids)
    for i, center in enumerate(ids):
        w = int(rng.integers(1, max_window + 1))
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((center, ids[j]))
    return pairs


def main():
    rng = np.random.default_rng(0)

    tokens = load_corpus()
    print(f"corpus tokens: {len(tokens)}")

    vocab = Vocabulary(tokens, min_count=1)
    print(f"vocabulary size: {len(vocab)}")
    print("most common:", [vocab.itos[i] for i in range(min(5, len(vocab)))])

    ids = vocab.encode(tokens)
    keep_prob = subsample_keep_prob(vocab.counts, vocab.total, threshold=1e-3)
    kept = subsample(ids, keep_prob, rng)
    print(f"tokens after subsampling: {len(kept)} / {len(ids)}")

    pairs = generate_pairs(kept, max_window=2, rng=rng)
    print(f"generated {len(pairs)} skip-gram pairs")
    for c, o in pairs[:8]:
        print(f"  center={vocab.itos[c]:<10} context={vocab.itos[o]}")


if __name__ == "__main__":
    main()
