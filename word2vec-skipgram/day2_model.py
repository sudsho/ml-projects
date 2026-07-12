"""
Day 2 of Word2Vec (Skip-Gram with Negative Sampling) from scratch.

Now that day 1 turns a corpus into (center, context) id pairs, this file adds
the model itself. Skip-gram keeps TWO embedding matrices:

    W_in  [V, D]  - the "input" / center-word vectors (what we ultimately keep)
    W_out [V, D]  - the "output" / context-word vectors used only for scoring

For a (center, context) pair the model wants the dot product v_center . u_context
to be large, and the dot products against a handful of random "negative" words to
be small. That is the negative-sampling objective, a cheap approximation to the
full softmax over the vocabulary.

Negatives are drawn from a unigram distribution raised to the 3/4 power, the
smoothing Mikolov et al. (2013) found works best - it lifts rare words and damps
very frequent ones relative to the raw unigram.

Everything stays in NumPy so the forward pass, the loss, and the gradients are
all visible. The training loop that consumes this lands on day 3.
"""

import numpy as np


def sigmoid(x):
    """Numerically stable logistic sigmoid, elementwise."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


class NoiseDistribution:
    """Unigram ** 0.75 sampler for drawing negative context words.

    Raising counts to the 3/4 power flattens the distribution: frequent words
    are still likely but rare words get a meaningful share of the mass. We cache
    the normalized probabilities and let numpy's Generator do the sampling.
    """

    def __init__(self, counts, power=0.75):
        weights = np.asarray(counts, dtype=np.float64) ** power
        self.probs = weights / weights.sum()
        self.vocab_size = len(self.probs)

    def sample(self, shape, rng):
        """Draw an array of negative word ids of the given shape."""
        return rng.choice(self.vocab_size, size=shape, p=self.probs)


class SkipGramNS:
    """Skip-gram with negative sampling, plain NumPy.

    Parameters live in two [V, D] matrices. The public surface is:
        forward_loss(...)  -> scalar loss for a minibatch (for monitoring)
        backward(...)      -> gradients wrt the rows touched by the batch
    Day 3 wires these into an SGD loop; keeping them separate makes the math
    easy to unit-test in isolation.
    """

    def __init__(self, vocab_size, dim=50, seed=0):
        self.vocab_size = vocab_size
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Small uniform init on the input side (the vectors we keep), zeros on
        # the output side - a common word2vec initialization choice.
        scale = 0.5 / dim
        self.W_in = rng.uniform(-scale, scale, size=(vocab_size, dim))
        self.W_out = np.zeros((vocab_size, dim))

    def _scores(self, centers, contexts):
        """Dot products v_center . u_context for aligned id arrays."""
        v = self.W_in[centers]            # [B, D]
        u = self.W_out[contexts]          # [B, D]
        return np.sum(v * u, axis=1)      # [B]

    def forward_loss(self, centers, pos_contexts, neg_contexts):
        """Negative-sampling loss averaged over a minibatch.

        centers       [B]        center word ids
        pos_contexts  [B]        the true context word for each center
        neg_contexts  [B, K]     K sampled negatives per center

        L = -log sigma(v.u_pos) - sum_k log sigma(-v.u_neg_k)
        """
        v = self.W_in[centers]                          # [B, D]
        u_pos = self.W_out[pos_contexts]                # [B, D]
        u_neg = self.W_out[neg_contexts]                # [B, K, D]

        pos_score = np.sum(v * u_pos, axis=1)           # [B]
        neg_score = np.einsum("bd,bkd->bk", v, u_neg)   # [B, K]

        eps = 1e-10
        pos_loss = -np.log(sigmoid(pos_score) + eps)
        neg_loss = -np.sum(np.log(sigmoid(-neg_score) + eps), axis=1)
        return float(np.mean(pos_loss + neg_loss))

    def backward(self, centers, pos_contexts, neg_contexts):
        """Gradients for one minibatch, returned as dense param-shaped arrays.

        Uses the clean logistic-loss gradients:
            d/dscore [-log sigma(pos)]  = sigma(pos) - 1
            d/dscore [-log sigma(-neg)] = sigma(neg)
        Gradients are accumulated with np.add.at because a word id can appear
        several times inside a single batch.
        """
        B = len(centers)
        v = self.W_in[centers]                          # [B, D]
        u_pos = self.W_out[pos_contexts]                # [B, D]
        u_neg = self.W_out[neg_contexts]                # [B, K, D]

        pos_score = np.sum(v * u_pos, axis=1)           # [B]
        neg_score = np.einsum("bd,bkd->bk", v, u_neg)   # [B, K]

        grad_pos = (sigmoid(pos_score) - 1.0)[:, None]  # [B, 1]
        grad_neg = sigmoid(neg_score)                   # [B, K]

        grad_in = np.zeros_like(self.W_in)
        grad_out = np.zeros_like(self.W_out)

        # Center vectors: pull from the positive, push from every negative.
        dv = grad_pos * u_pos + np.einsum("bk,bkd->bd", grad_neg, u_neg)
        np.add.at(grad_in, centers, dv)

        # Positive context vectors.
        np.add.at(grad_out, pos_contexts, grad_pos * v)

        # Negative context vectors, one update per (batch, k) slot.
        du_neg = grad_neg[:, :, None] * v[:, None, :]   # [B, K, D]
        np.add.at(grad_out, neg_contexts, du_neg)

        return grad_in / B, grad_out / B


def _numeric_gradient_check():
    """Finite-difference sanity check on a tiny random problem.

    Confirms the analytic center-vector gradient matches a numerical estimate,
    which is the part most likely to have an einsum bug.
    """
    rng = np.random.default_rng(1)
    model = SkipGramNS(vocab_size=12, dim=6, seed=2)
    centers = rng.integers(0, 12, size=8)
    pos = rng.integers(0, 12, size=8)
    neg = rng.integers(0, 12, size=(8, 4))

    grad_in, _ = model.backward(centers, pos, neg)

    # Check a single parameter entry against a central difference.
    i, j = int(centers[0]), 0
    eps = 1e-5
    model.W_in[i, j] += eps
    plus = model.forward_loss(centers, pos, neg)
    model.W_in[i, j] -= 2 * eps
    minus = model.forward_loss(centers, pos, neg)
    model.W_in[i, j] += eps
    numeric = (plus - minus) / (2 * eps)

    analytic = grad_in[i, j]
    print(f"grad check  analytic={analytic:+.6f}  numeric={numeric:+.6f}")
    assert abs(analytic - numeric) < 1e-4, "gradient mismatch"
    print("gradient check passed")


def main():
    rng = np.random.default_rng(0)
    vocab_size, dim = 30, 16

    # A fake unigram count vector standing in for day 1's vocab.counts.
    counts = rng.integers(1, 100, size=vocab_size)
    noise = NoiseDistribution(counts, power=0.75)

    model = SkipGramNS(vocab_size, dim=dim, seed=0)
    print(f"W_in {model.W_in.shape}, W_out {model.W_out.shape}")

    # One synthetic minibatch: centers, their positive contexts, K negatives each.
    batch = 8
    k = 5
    centers = rng.integers(0, vocab_size, size=batch)
    pos_contexts = rng.integers(0, vocab_size, size=batch)
    neg_contexts = noise.sample((batch, k), rng)

    loss = model.forward_loss(centers, pos_contexts, neg_contexts)
    grad_in, grad_out = model.backward(centers, pos_contexts, neg_contexts)
    print(f"initial batch loss: {loss:.4f}")
    print(f"grad_in norm: {np.linalg.norm(grad_in):.4f}  "
          f"grad_out norm: {np.linalg.norm(grad_out):.4f}")

    _numeric_gradient_check()


if __name__ == "__main__":
    main()
