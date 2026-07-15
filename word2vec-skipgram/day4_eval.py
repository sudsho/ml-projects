"""
Day 4 of Word2Vec (Skip-Gram with Negative Sampling) from scratch.

Days 1-3 built the pipeline, the model, and the training loop. Day 4 is about
looking at what the embeddings actually learned, using three classic probes:

  * nearest neighbours      - for a query word, the most cosine-similar words
  * word analogies          - the "king - man + woman ~= queen" vector arithmetic,
                              solved by cosine against (b - a + c)
  * a 2-D t-SNE map         - project the high-dim vectors down so neighbourhoods
                              are visible; written from scratch in NumPy so the
                              affinity/gradient machinery stays inspectable

On the tiny toy corpus the numbers are noisy, but the plumbing is exactly what
you would run on a real embedding matrix. matplotlib is optional: if it is not
installed we still dump the 2-D coordinates to a text file.
"""

import numpy as np

from day3_train import train, cosine_neighbours


def _unit_rows(mat):
    """L2-normalize each row so dot products become cosine similarities."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


def analogy(model, vocab, a, b, c, topn=3):
    """Solve a : b :: c : ? by cosine against the vector (b - a + c).

    Returns the top ranked candidate words, excluding the three inputs so we
    don't trivially return one of them.
    """
    for w in (a, b, c):
        if w not in vocab.stoi:
            return []
    unit = _unit_rows(model.W_in)
    target = unit[vocab.stoi[b]] - unit[vocab.stoi[a]] + unit[vocab.stoi[c]]
    target /= max(np.linalg.norm(target), 1e-8)

    sims = unit @ target
    banned = {vocab.stoi[a], vocab.stoi[b], vocab.stoi[c]}
    out = []
    for idx in np.argsort(-sims):
        if idx in banned:
            continue
        out.append((vocab.itos[idx], float(sims[idx])))
        if len(out) >= topn:
            break
    return out


def _pairwise_affinities(X, perplexity, rng, n_steps=50):
    """High-dimensional p_{j|i} with a per-point sigma tuned to the perplexity.

    For each row we binary-search the Gaussian bandwidth so the conditional
    distribution has the requested perplexity (2**entropy), then symmetrize
    into the joint P used by t-SNE.
    """
    n = X.shape[0]
    sq = np.sum(X * X, axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(dist, np.inf)

    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = 1e-20, np.inf, 1.0
        for _ in range(n_steps):
            row = np.exp(-dist[i] * beta)
            row[i] = 0.0
            s = row.sum()
            if s <= 0:
                beta /= 2.0
                continue
            p = row / s
            entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
            if entropy < target:
                hi = beta
                beta = (beta + lo) / 2.0
            else:
                lo = beta
                beta = beta * 2.0 if hi == np.inf else (beta + hi) / 2.0
        row = np.exp(-dist[i] * beta)
        row[i] = 0.0
        P[i] = row / max(row.sum(), 1e-12)

    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def tsne(X, dim=2, perplexity=5.0, n_iter=300, lr=200.0, seed=0):
    """Minimal Barnes-Hut-free t-SNE in NumPy for small embedding sets.

    Uses early exaggeration for the first 100 steps and momentum on the update,
    the two tricks that make plain t-SNE converge to readable maps.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    P = _pairwise_affinities(X, min(perplexity, (n - 1) / 3.0), rng)
    P *= 4.0  # early exaggeration

    Y = rng.normal(0, 1e-4, size=(n, dim))
    velocity = np.zeros_like(Y)

    for it in range(n_iter):
        sq = np.sum(Y * Y, axis=1)
        num = 1.0 / (1.0 + sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)

        grad = np.zeros_like(Y)
        PQ = (P - Q) * num
        for i in range(n):
            grad[i] = 4.0 * np.sum((PQ[i][:, None]) * (Y[i] - Y), axis=0)

        momentum = 0.5 if it < 50 else 0.8
        velocity = momentum * velocity - lr * grad
        Y += velocity
        Y -= Y.mean(axis=0)

        if it == 100:
            P /= 4.0  # stop early exaggeration
    return Y


def save_map(coords, words, path="tsne_map"):
    """Plot the 2-D map if matplotlib is available, else dump coordinates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)
        for (x, y), w in zip(coords, words):
            plt.annotate(w, (x, y), fontsize=7, alpha=0.75)
        plt.title("word2vec skip-gram embeddings (t-SNE)")
        plt.tight_layout()
        plt.savefig(f"{path}.png", dpi=140)
        plt.close()
        print(f"wrote {path}.png")
    except Exception as exc:  # matplotlib missing or headless issue
        with open(f"{path}.txt", "w", encoding="utf-8") as fh:
            for (x, y), w in zip(coords, words):
                fh.write(f"{w}\t{x:.4f}\t{y:.4f}\n")
        print(f"matplotlib unavailable ({exc}); wrote {path}.txt")


def main():
    # A short training run - day 4 is about evaluation, not squeezing loss.
    model, vocab = train(n_epochs=160, log_every=160, probes=("king", "queen", "paris"))

    print("\n=== nearest neighbours ===")
    for w in ("king", "queen", "paris", "capital"):
        nbrs = cosine_neighbours(model, vocab, w, topn=5)
        if nbrs:
            shown = ", ".join(f"{n}:{s:+.2f}" for n, s in nbrs)
            print(f"  {w:<8} -> {shown}")

    print("\n=== analogies ===")
    # geography analogies over words the toy corpus actually contains
    for a, b, c in [("france", "paris", "germany"), ("paris", "france", "berlin")]:
        ans = analogy(model, vocab, a, b, c, topn=3)
        if ans:
            shown = ", ".join(f"{n}:{s:+.2f}" for n, s in ans)
            print(f"  {a} : {b} :: {c} : ?  ->  {shown}")

    print("\n=== t-SNE projection ===")
    emb = model.W_in
    coords = tsne(emb, dim=2, perplexity=5.0, n_iter=300, seed=0)
    words = [vocab.itos[i] for i in range(len(vocab))]
    print(f"projected {emb.shape} -> {coords.shape}")
    save_map(coords, words, path="tsne_map")


if __name__ == "__main__":
    main()
