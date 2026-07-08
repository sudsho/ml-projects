# Neural Machine Translation with Seq2Seq + Attention

An encoder-decoder LSTM with Bahdanau (additive) attention built from scratch in
PyTorch for word-level translation: a parallel-corpus data pipeline, a
bidirectional-LSTM encoder, an attention decoder driven one step at a time,
teacher-forced training with masked cross-entropy, then greedy/beam decoding,
BLEU evaluation, and attention heatmaps.

## Overview

The model reads a source sentence, encodes it into a sequence of context-aware
hidden states, and generates the target sentence one token at a time. At every
decode step an attention module scores the current decoder state against all
encoder outputs and builds a context vector, so the decoder can look back at the
relevant source words instead of relying on a single fixed summary vector. The
corpus is a tiny toy English-French set, which keeps the focus on the mechanics -
packing, masking, attention, teacher forcing, and beam search - rather than on
data wrangling.

## Layout

| Day | File | What it covers |
|-----|------|----------------|
| 1 | `day1_data.py` | Parallel-corpus pipeline: tokenization, per-side vocab with `<pad>/<sos>/<eos>/<unk>`, padded batches and source lengths |
| 2 | `day2_encoder_attention.py` | Bidirectional-LSTM encoder and the Bahdanau additive-attention module with padding masks |
| 3 | `day3_decoder_training.py` | Attention decoder (LSTMCell) and the teacher-forced training loop with masked cross-entropy and gradient clipping |
| 4 | `day4_decoding_bleu.py` | Greedy and beam-search decoding, corpus BLEU, attention-heatmap visualization |

## Method notes

- **Packing.** The encoder packs each padded batch by its true source lengths, so
  the LSTM never spends state or computation on `<pad>` steps. Decoder-side
  targets are wrapped in `<sos> ... <eos>` so every step supplies a next-token
  target.
- **Bahdanau attention.** An additive score `v^T tanh(W_dec h + W_enc e_t)`
  between the decoder state and each encoder output, softmaxed into a weighting.
  Padded source positions are masked to `-inf` *before* the softmax, otherwise
  the alignments smear onto padding and quality collapses.
- **Step-at-a-time decoder.** Because the attention context for each step depends
  on that step's hidden state, the decoder runs on an `LSTMCell` one token at a
  time. The readout sees `[hidden ; context ; embedding]` so the vocabulary
  distribution reads the aligned source words directly.
- **Teacher forcing.** Training feeds the gold previous token most of the time
  but occasionally feeds the model's own argmax, so it learns to recover from its
  own mistakes rather than only ever seeing perfect history.
- **Decoding.** Greedy feeds each argmax back in until `<eos>`; beam search keeps
  the `k` best hypotheses and length-normalises (`score / len^alpha`) before the
  final pick so it does not just prefer the shortest sentence.
- **BLEU.** Corpus-level modified n-gram precision with add-1 smoothing on the
  higher orders and the brevity penalty, accumulated across the corpus before
  dividing rather than averaged per sentence.

## Running

```bash
python day1_data.py                # vocab sizes and a decoded padded batch
python day2_encoder_attention.py   # encoder/attention shapes and a masking check
python day3_decoder_training.py    # a short overfit run; loss should drop clearly
python day4_decoding_bleu.py       # greedy/beam translations, corpus BLEU, heatmap
```

Each file has a `__main__` smoke test that runs on CPU. Day 4 overfits the toy
corpus and then reaches BLEU 1.0 on it, which confirms the encoder, attention,
decoder, masking, and decoding are all wired together correctly.

## Stack

PyTorch (matplotlib optional, only for the attention heatmap). Generated
outputs (`samples/`, `*.png`) are kept out of git.
