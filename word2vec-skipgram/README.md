# Word2Vec Skip-Gram with Negative Sampling from Scratch

A Word2Vec skip-gram model built from scratch in NumPy on a small corpus: text
preprocessing with frequency-based subsampling and dynamic-window (center,
context) pairs, input/output embedding matrices trained with negative sampling
against a unigram^(3/4) noise distribution, then embedding evaluation with
cosine-similarity neighbours, analogy arithmetic, and a from-scratch t-SNE map.

## Overview

Skip-gram learns word vectors by predicting a word's context from the word
itself. Rather than a full softmax over the vocabulary, it uses negative
sampling: for each true (center, context) pair we push the center vector toward
the context vector and away from a handful of random "noise" words. Two
embedding matrices are trained - `W_in` (the vectors we keep) and `W_out` (the
context vectors used only for scoring). The corpus is a tiny toy set with a
geography/animals theme, which keeps the focus on the mechanics - subsampling,
dynamic windows, the noise distribution, and the negative-sampling gradients -
rather than on data at scale.

## Layout

| Day | File | What it covers |
|-----|------|----------------|
| 1 | `day1_data.py` | Tokenization, frequency-filtered vocabulary, Mikolov subsampling of frequent words, and dynamic-window skip-gram (center, context) pair generation |
| 2 | `day2_model.py` | Skip-gram model with `W_in`/`W_out` embeddings, the negative-sampling loss over a unigram^0.75 noise distribution, and analytic gradients with a finite-difference check |
| 3 | `day3_train.py` | Minibatch SGD training loop with per-epoch pair resampling, fresh negatives per batch, linear LR decay, and periodic nearest-neighbour sanity checks |
| 4 | `day4_eval.py` | Cosine-similarity neighbours, `b - a + c` analogy arithmetic, and a NumPy t-SNE projection of the learned embeddings |

## Method notes

- **Subsampling.** Frequent words are dropped with probability
  `1 - sqrt(t / f)` (Mikolov's rule), which thins out uninformative tokens like
  "the" and lets the windows see more content words.
- **Dynamic window.** For each center word the effective window is drawn
  uniformly up to `max_window`, so nearer context words are sampled more often -
  a cheap distance weighting baked into the sampling rather than the loss.
- **Negative sampling.** The loss is
  `-log sigma(v.u_pos) - sum_k log sigma(-v.u_neg_k)`. Negatives are drawn from
  the unigram distribution raised to the 3/4 power, which lifts rare words and
  damps very frequent ones relative to the raw unigram.
- **Gradients.** Written out by hand from the logistic-loss derivatives
  (`sigma(pos) - 1` and `sigma(neg)`) and accumulated with `np.add.at` because a
  word id can appear several times in one batch. A numeric gradient check in
  `day2_model.py` guards the center-vector einsum.
- **Fresh view each epoch.** Subsampling and the dynamic windows are re-rolled
  every epoch, so the model sees a slightly different, noisier corpus each pass -
  what the original C implementation does.
- **t-SNE.** A compact from-scratch t-SNE: per-point Gaussian bandwidths tuned to
  a target perplexity by binary search, symmetrized joint affinities, then
  gradient descent with early exaggeration and momentum. `matplotlib` is optional -
  without it the 2-D coordinates are dumped to a text file.

## Running

```bash
python day1_data.py    # vocab, subsampling keep-probs, a few sample pairs
python day2_model.py   # forward loss on a batch and the numeric gradient check
python day3_train.py   # training loop; nearest neighbours should sharpen over epochs
python day4_eval.py    # neighbours, analogies, and the t-SNE map (tsne_map.png)
```

## Notes

On such a tiny corpus the cosine similarities saturate quickly and the analogy
results are noisy - the point here is the machinery, which transfers directly to
a real corpus by swapping in `day1_data.load_corpus`.
