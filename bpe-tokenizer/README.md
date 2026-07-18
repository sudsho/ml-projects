# Byte-Pair Encoding Tokenizer from Scratch

A subword BPE tokenizer built in pure Python, no external libraries. It trains a
merge table from a corpus by greedily fusing the most frequent adjacent symbol
pair, then encodes/decodes new text by replaying those merges - the same
algorithm behind the tokenizers used by GPT and most modern LLMs, stripped down
so every step is visible.

## Idea

BPE sits between two extremes. Character tokenization never has out-of-vocabulary
words but makes sequences long; word tokenization keeps sequences short but the
vocabulary explodes and rare words fall off the edge. BPE interpolates: start
from characters and repeatedly merge the most frequent adjacent pair, learning
subword units ("low", "est</w>", "er</w>") that are common enough to be worth a
vocabulary slot. Frequent words collapse to a single token; rare words still
decompose into known pieces, so nothing is ever unrepresentable.

## Layout

- `day1_data.py` - corpus preprocessing, per-word frequency table, and the
  initial character vocabulary with `</w>` end-of-word markers.
- `day2_train.py` - the training loop: count adjacent pairs weighted by word
  frequency, greedily merge the most frequent, record the ordered merge table,
  track vocabulary growth and shrinking symbol counts.
- `day3_encode.py` - encoder/decoder that replays the merges in rank order,
  round-trip correctness tests, and the vocab-size vs sequence-length sweep.

## Key design choices

- **End-of-word marker (`</w>`).** Merges never cross a word boundary, and the
  marker lets the decoder tell a word-final suffix from the same letters
  mid-word, so decoding is exact.
- **Frequency-weighted pair counts.** A pair is scored by how many token
  occurrences contain it, not how many distinct words, which is what makes BPE
  latch onto genuinely common stems instead of rare noise.
- **Deterministic tie-breaking.** Ties resolve by (count, then lexicographic
  order) so the same corpus always yields the same merge table - reproducible
  tokenization.
- **Rank-ordered encoding.** Encoding a new word applies the lowest-rank
  (earliest-learned) merge available at each step, reproducing exactly the
  segmentation training would have produced.

## Running

```bash
python day1_data.py     # corpus, initial vocabulary, baseline symbol count
python day2_train.py    # learn the merge table, watch vocab grow / symbols fall
python day3_encode.py   # encode/decode, round-trip tests, tradeoff curve
```

## Results

On the built-in demo corpus, 20 merges take an unseen word like `lowest` down to
two tokens (`low` + `est</w>`) and cut the corpus token count from 229 (pure
characters) to 174 while the vocabulary grows from 26 to 46. Every example
round-trips: `decode(encode(text))` reproduces the normalized input exactly. The
vocab-size vs sequence-length curve printed by `day3_encode.py` shows the classic
diminishing-returns knee - the first handful of merges buy most of the length
reduction, later ones far less.
