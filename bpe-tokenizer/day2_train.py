"""
Day 2 of the BPE tokenizer: the training loop that learns the merge table.

Day 1 turned the corpus into {symbol_tuple: frequency}. Training is now a greedy
loop: count every adjacent symbol pair across the corpus (weighted by word
frequency), merge the single most frequent pair into one new symbol everywhere it
occurs, and record that merge. Repeat until we hit the target number of merges or
no pair repeats. The ordered list of merges *is* the learned model - day 3's
encoder just replays these merges in order on new text.

Two details that matter:
  * Pairs are counted with word frequency as the weight, so a pair inside a word
    that appears 200 times counts 200 times. That is what makes BPE latch onto
    common subword stems ('est</w>', 'low') instead of rare noise.
  * Ties are broken deterministically (highest count, then lexicographic) so the
    same corpus always yields the same merge table - important for reproducible
    tokenization.
"""

from collections import Counter

from day1_data import END_OF_WORD, load_corpus, initial_vocabulary, corpus_stats


def count_pairs(corpus):
    """Frequency of every adjacent symbol pair across the whole corpus.

    Each word contributes its pairs `freq` times, so a pair is scored by how many
    token occurrences (not distinct words) contain it.
    """
    pairs = Counter()
    for symbols, freq in corpus.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(corpus, pair):
    """Return a new corpus with every occurrence of `pair` fused into one symbol.

    Walk each word left to right; whenever the two target symbols sit adjacent,
    emit the concatenated symbol and skip ahead by two, otherwise copy one symbol
    and advance by one. Frequencies carry over unchanged - only the symbol
    sequences shrink.
    """
    a, b = pair
    merged = a + b
    new_corpus = {}
    for symbols, freq in corpus.items():
        out = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_corpus[tuple(out)] = freq
    return new_corpus


def best_pair(pairs):
    """Most frequent pair, ties broken lexicographically for determinism."""
    return max(pairs, key=lambda p: (pairs[p], p))


def train_bpe(corpus, num_merges, verbose=False):
    """Learn up to `num_merges` merges greedily; return (merges, final corpus).

    `merges` is the ordered list of merged pairs - the model. We also track how
    the vocabulary grows and how the total symbol count falls, since that
    shrinking sequence length is the payoff BPE is after.
    """
    corpus = dict(corpus)
    merges = []
    vocab = set(initial_vocabulary(corpus))
    history = []

    for step in range(num_merges):
        pairs = count_pairs(corpus)
        if not pairs:
            break
        pair = best_pair(pairs)
        if pairs[pair] < 2:
            # Nothing repeats anymore; further merges would just memorize noise.
            break

        corpus = merge_pair(corpus, pair)
        merges.append(pair)
        vocab.add(pair[0] + pair[1])

        total_symbols = sum(len(s) * f for s, f in corpus.items())
        history.append((step + 1, pair, pairs[pair], len(vocab), total_symbols))
        if verbose:
            a, b = pair
            print(f"  merge {step + 1:2d}: ({a!r}, {b!r}) "
                  f"count={pairs[pair]:3d}  vocab={len(vocab):3d}  "
                  f"symbols={total_symbols:3d}")

    return merges, corpus, history


def merge_table(merges):
    """Map each merged pair to its rank (application order) for the encoder."""
    return {pair: rank for rank, pair in enumerate(merges)}


if __name__ == "__main__":
    corpus = load_corpus()
    base = corpus_stats(corpus)
    print(f"before training: vocab={len(initial_vocabulary(corpus))}  "
          f"symbols={base['total_symbols_no_merges']}\n")

    print("training merges:")
    merges, final_corpus, history = train_bpe(corpus, num_merges=20, verbose=True)

    print(f"\nlearned {len(merges)} merges")
    print("first few merges (in order):")
    for rank, pair in enumerate(merges[:8]):
        print(f"  {rank}: {pair[0]!r} + {pair[1]!r} -> {(pair[0] + pair[1])!r}")

    if history:
        _, _, _, final_vocab, final_symbols = history[-1]
        shrink = 100 * (1 - final_symbols / base["total_symbols_no_merges"])
        print(f"\nsymbol count {base['total_symbols_no_merges']} -> {final_symbols} "
              f"({shrink:.1f}% shorter), vocabulary grew to {final_vocab}")
