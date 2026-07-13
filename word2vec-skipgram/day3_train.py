"""
Day 3 of Word2Vec (Skip-Gram with Negative Sampling) from scratch.

Days 1 and 2 gave us the two halves that day 3 finally connects:

    day1_data   -> corpus -> vocab, subsampling, (center, context) pairs
    day2_model  -> SkipGramNS with forward_loss / backward and a noise sampler

Here we run the actual optimization. The loop is plain minibatch SGD over the
skip-gram pairs, with three details that matter for word2vec quality:

  * pairs are regenerated every epoch, so the dynamic window and the frequent-
    word subsampling are re-rolled each pass rather than frozen once. That gives
    the model a slightly different, noisier view of the corpus each epoch, which
    is exactly what Mikolov's implementation does.
  * K negatives are drawn per positive from the unigram**0.75 noise distribution,
    freshly for every batch.
  * a linear learning-rate decay from lr_start down to a small floor, the schedule
    the original C tool uses.

Every few epochs we print the nearest neighbours of a couple of probe words by
cosine similarity - the cheapest way to see whether anything is being learned on
such a tiny corpus.
"""

import numpy as np

from day1_data import (
    load_corpus,
    Vocabulary,
    subsample_keep_prob,
    subsample,
    generate_pairs,
)
from day2_model import SkipGramNS, NoiseDistribution


def iterate_minibatches(pairs, batch_size, rng):
    """Shuffle the pair list and yield (centers, contexts) id arrays."""
    order = rng.permutation(len(pairs))
    pairs = np.asarray(pairs)[order]
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        yield chunk[:, 0], chunk[:, 1]


def cosine_neighbours(model, vocab, word, topn=5):
    """Return the topn nearest words to `word` by cosine similarity of W_in."""
    if word not in vocab.stoi:
        return []
    emb = model.W_in
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.clip(norms, 1e-8, None)
    q = unit[vocab.stoi[word]]
    sims = unit @ q
    order = np.argsort(-sims)
    out = []
    for idx in order:
        if idx == vocab.stoi[word]:
            continue
        out.append((vocab.itos[idx], float(sims[idx])))
        if len(out) >= topn:
            break
    return out


def linear_lr(epoch, n_epochs, lr_start, lr_floor):
    """Linearly decay the learning rate, clamped at lr_floor."""
    frac = epoch / max(1, n_epochs)
    return max(lr_floor, lr_start * (1.0 - frac))


def train(
    corpus_path=None,
    dim=50,
    n_epochs=400,
    batch_size=16,
    n_negatives=5,
    max_window=3,
    lr_start=0.5,
    lr_floor=1e-4,
    subsample_t=1e-3,
    seed=0,
    log_every=80,
    probes=("king", "capital", "the"),
):
    """Full skip-gram training run; returns the fitted model and its vocab."""
    rng = np.random.default_rng(seed)

    tokens = load_corpus(corpus_path)
    vocab = Vocabulary(tokens, min_count=1)
    ids = vocab.encode(tokens)
    keep_prob = subsample_keep_prob(vocab.counts, vocab.total, threshold=subsample_t)

    model = SkipGramNS(len(vocab), dim=dim, seed=seed)
    noise = NoiseDistribution(vocab.counts, power=0.75)

    print(f"vocab={len(vocab)}  raw_tokens={len(ids)}  dim={dim}")

    for epoch in range(n_epochs):
        # Re-roll subsampling and dynamic windows so every epoch sees a fresh
        # sample of the corpus, then reshape into training pairs.
        kept = subsample(ids, keep_prob, rng)
        pairs = generate_pairs(kept, max_window=max_window, rng=rng)
        if not pairs:
            continue

        lr = linear_lr(epoch, n_epochs, lr_start, lr_floor)
        epoch_loss = 0.0
        n_batches = 0

        for centers, contexts in iterate_minibatches(pairs, batch_size, rng):
            negatives = noise.sample((len(centers), n_negatives), rng)

            epoch_loss += model.forward_loss(centers, contexts, negatives)
            grad_in, grad_out = model.backward(centers, contexts, negatives)

            # Plain SGD step; grads already divided by batch size in backward().
            model.W_in -= lr * grad_in
            model.W_out -= lr * grad_out
            n_batches += 1

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            avg = epoch_loss / max(1, n_batches)
            print(f"epoch {epoch:3d}  lr={lr:.4f}  pairs={len(pairs):5d}  loss={avg:.4f}")
            for w in probes:
                nbrs = cosine_neighbours(model, vocab, w, topn=3)
                if nbrs:
                    shown = ", ".join(f"{n}:{s:+.2f}" for n, s in nbrs)
                    print(f"    {w:<8} -> {shown}")

    return model, vocab


def main():
    model, vocab = train()
    print("\nfinal embedding matrix:", model.W_in.shape)
    # Quick smoke check: cosine similarity is symmetric and self-sim is ~1.
    unit = model.W_in / np.clip(np.linalg.norm(model.W_in, axis=1, keepdims=True), 1e-8, None)
    self_sim = float(unit[0] @ unit[0])
    assert abs(self_sim - 1.0) < 1e-6, "unit-normalized self similarity should be 1"
    print(f"self-similarity sanity: {self_sim:.6f}")


if __name__ == "__main__":
    main()
