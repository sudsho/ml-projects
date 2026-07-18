"""
Day 3 of the BPE tokenizer: encode/decode with the learned merges, verify
round-trip correctness, and measure the vocabulary-size vs sequence-length
tradeoff that motivates BPE in the first place.

Day 2 produced an ordered merge list - the model. Encoding a *new* word means
replaying those merges on it: start from the raw character sequence (plus the
end-of-word marker) and, at each step, apply the learnable merge with the lowest
rank that still appears in the word. Lowest rank = earliest learned = most
frequent in the training corpus, so this reproduces exactly the segmentation
training would have produced for that word. Repeat until no known merge applies.

Decoding is trivial: concatenate the symbols and strip the end-of-word marker
back to a space. Because every merge is just string concatenation, the surface
text is always recoverable - that is the round-trip guarantee we test below.
"""

from day1_data import END_OF_WORD, normalize, word_to_symbols, load_corpus, corpus_stats
from day2_train import train_bpe, merge_table


def encode_word(word, ranks):
    """Segment a single word into subword symbols using the merge ranks.

    `ranks` maps (a, b) -> application order. We repeatedly find the adjacent
    pair present in the word with the smallest rank and merge it, mirroring the
    order training learned them in. When no adjacent pair is known we are done.
    """
    symbols = list(word_to_symbols(word))
    if len(symbols) == 1:
        return symbols

    while True:
        # Find the best (lowest-rank) merge available among current adjacencies.
        best_rank = None
        best_i = None
        for i in range(len(symbols) - 1):
            rank = ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_i = i

        if best_i is None:
            break

        # Fuse the winning pair and restart the scan on the shortened sequence.
        symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]

    return symbols


def encode(text, ranks):
    """Encode a whole text into a flat list of subword tokens."""
    tokens = []
    for word in normalize(text).split():
        tokens.extend(encode_word(word, ranks))
    return tokens


def decode(tokens):
    """Invert encode: glue symbols back and turn end markers into spaces.

    Each token already carries its own characters, and the end-of-word marker
    records where a word finished, so joining and replacing the marker with a
    space rebuilds the normalized surface string exactly.
    """
    text = "".join(tokens)
    return text.replace(END_OF_WORD, " ").strip()


def round_trip_ok(text, ranks):
    """True when encode->decode reproduces the normalized input word for word."""
    return decode(encode(text, ranks)) == normalize(text)


def sequence_length(text, ranks):
    """Number of tokens the encoder emits for `text` under the current model."""
    return len(encode(text, ranks))


def vocab_vs_length(corpus, text, merge_counts):
    """Sweep the number of merges and record (vocab_size, tokens_emitted).

    This is the central BPE tradeoff: every merge adds one vocabulary entry but
    shortens sequences. Early merges buy a lot of length reduction cheaply; later
    ones show diminishing returns. Returning the curve lets day-3's report (or a
    plot) show exactly where that knee sits for this corpus.
    """
    curve = []
    for k in merge_counts:
        merges, _, _ = train_bpe(corpus, num_merges=k)
        ranks = merge_table(merges)
        base_vocab = len(set(s for symbols in corpus for s in symbols))
        vocab_size = base_vocab + len(merges)
        curve.append((k, vocab_size, sequence_length(text, ranks)))
    return curve


if __name__ == "__main__":
    corpus = load_corpus()
    merges, _, _ = train_bpe(corpus, num_merges=20)
    ranks = merge_table(merges)

    samples = [
        "lowest",          # unseen word built from seen stems: 'low' + 'est'
        "newest widest",
        "the tokenizer",
        "slowness",        # 's' + learned 'low' + 'ness' pieces
    ]

    print("encoding examples:")
    for s in samples:
        toks = encode(s, ranks)
        ok = round_trip_ok(s, ranks)
        print(f"  {s!r:22s} -> {toks}")
        print(f"  {'':22s}    round-trip {'OK' if ok else 'FAILED'}, "
              f"{len(toks)} tokens")

    # A blunt correctness check across the whole training corpus.
    corpus_text = " ".join(
        "".join(sym for sym in symbols if sym != END_OF_WORD)
        for symbols in corpus
    )
    all_ok = round_trip_ok(corpus_text, ranks)
    print(f"\nfull-corpus round-trip: {'OK' if all_ok else 'FAILED'}")

    print("\nvocab-size vs sequence-length tradeoff (encoding the corpus text):")
    curve = vocab_vs_length(corpus, corpus_text, merge_counts=[0, 2, 4, 6, 8, 12, 16, 20])
    print(f"  {'merges':>7} {'vocab':>7} {'tokens':>7}")
    for k, vocab_size, n_tokens in curve:
        print(f"  {k:7d} {vocab_size:7d} {n_tokens:7d}")

    baseline = corpus_stats(corpus)["total_symbols_no_merges"]
    final_tokens = curve[-1][2]
    print(f"\nno-merge symbol count vs {curve[-1][0]}-merge token count: "
          f"{baseline} -> {final_tokens}")
