# Character-Level Transformer Language Model

A decoder-only transformer built from scratch in PyTorch and trained on the
tiny-shakespeare corpus: character tokenization, masked multi-head self-attention,
the full pre-norm block, then training and autoregressive sampling.

## Overview

The model reads text one character at a time and is trained to predict the next
character at every position. There is no subword tokenizer - the vocabulary is
just the ~65 distinct characters in the corpus - which keeps the focus on the
transformer itself: how attention mixes context, how the blocks stack, and how
sampling turns next-character probabilities back into text.

## Layout

| Day | File | What it covers |
|-----|------|----------------|
| 1 | `day1_data.py` | Character vocab, train/val split, batched context-window data loader |
| 2 | `day2_attention.py` | Scaled dot-product attention and multi-head self-attention with a causal mask |
| 3 | `day3_model.py` | Pre-norm transformer block (attention + MLP) and the full model |
| 4 | `day4_train.py` | AdamW + warmup/cosine LR training loop, autoregressive sampling, loss curve |

## Method notes

- **Tokenization.** Characters are mapped to a contiguous integer range in
  sorted order. A training example is a window `x = tokens[i:i+T]` paired with
  the same window shifted by one, `y = tokens[i+1:i+T+1]`, so every position
  supplies a next-character target.
- **Causal attention.** Each head scores queries against keys, scales by
  `1/sqrt(d)`, and masks the strict upper triangle to `-inf` before softmax, so
  position `t` can only attend to positions `<= t` - the model never sees the
  character it is predicting.
- **Pre-norm blocks.** LayerNorm is applied to the *input* of each sub-layer,
  `x = x + attn(ln1(x))` then `x = x + mlp(ln2(x))`. The clean residual path is
  what lets the blocks stack without the training going unstable.
- **Schedule.** AdamW with a short linear warmup followed by cosine decay to a
  small floor; gradients are clipped to norm 1.0. Train/val loss is averaged
  over many batches at each checkpoint rather than read off a single noisy batch.
- **Sampling.** Generation is autoregressive: take the last position's logits,
  divide by a temperature, softmax, sample one character, append, and crop the
  context to the last `block_size` tokens since the positional embedding only
  covers that many positions.

## Running

```bash
python day1_data.py        # vocab size, split sizes, a decoded batch
python day2_attention.py   # attention shapes and a causal-mask check
python day3_model.py       # parameter count and an untrained-loss sanity check
python day4_train.py       # train; writes samples/loss_curve.png and samples/sample.txt
```

Each file has a `__main__` smoke test that runs on CPU, so the modules can be
exercised in isolation before a full training run.

## Stack

PyTorch (matplotlib optional, only for the loss curve). The corpus download
(`input.txt`) and generated outputs (`samples/`) are kept out of git.
