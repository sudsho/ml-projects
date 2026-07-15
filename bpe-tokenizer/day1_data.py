"""
Day 1 of the Byte-Pair Encoding (BPE) tokenizer from scratch.

Before any merges happen we need the raw material BPE trains on: a corpus split
into words, a frequency table over those words, and a representation of each word
as a sequence of atomic symbols. BPE never merges across a word boundary, so we
mark the end of every word with a special end-of-word token ("</w>"). That marker
does double duty: it stops merges from leaking across whitespace and it lets the
decoder tell "est " (a word-final suffix) apart from "est" sitting mid-word.

Everything here is plain Python so the mechanics stay visible. Day 2 counts
adjacent symbol pairs and greedily merges the most frequent one; today we just
lay down the corpus, the word counts, and the initial character vocabulary.
"""

import re
from collections import Counter


END_OF_WORD = "</w>"


# A tiny built-in corpus so the script runs with no downloads. The repeated word
# stems (low / lower / newest / widest) are the classic BPE teaching example:
# they share subword pieces that merges should eventually discover.
SAMPLE_TEXT = """
low low low low low lower lower newest newest newest newest newest newest widest widest widest
the tokenizer learns subword units by merging frequent adjacent symbol pairs
byte pair encoding starts from characters and grows a vocabulary of merges
lower cost means the newest widest model still fits in memory on one machine
""".strip()


def normalize(text):
    """Lowercase and collapse any run of whitespace to single spaces.

    We keep only letters here to keep the demo readable. A production tokenizer
    would operate on raw bytes so nothing is ever out-of-vocabulary, but the
    merge logic is identical either way.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_frequencies(text):
    """Return a Counter mapping each whitespace-delimited word to its count."""
    return Counter(normalize(text).split())


def word_to_symbols(word):
    """Split a word into its initial symbol sequence: characters + end marker.

    'low' -> ('l', 'o', 'w', '</w>'). Using a tuple makes the sequence hashable
    so the training loop can key merge counts off whole words cheaply.
    """
    return tuple(word) + (END_OF_WORD,)


def build_corpus(text):
    """Frequency table keyed by the symbol-tuple form of each unique word.

    This is the object BPE training actually consumes: {symbols: count}. Keying
    by the tuple (rather than the raw string) means that after a merge rewrites a
    word's symbols we can keep carrying its frequency along unchanged.
    """
    corpus = {}
    for word, freq in word_frequencies(text).items():
        corpus[word_to_symbols(word)] = freq
    return corpus


def initial_vocabulary(corpus):
    """Every distinct symbol present before any merge, sorted for determinism.

    For byte-level BPE this is a fixed set of 256 byte values; here it is just
    the characters that actually occur plus the end-of-word marker.
    """
    vocab = set()
    for symbols in corpus:
        vocab.update(symbols)
    return sorted(vocab)


def corpus_stats(corpus):
    """A few sanity numbers: unique words, total tokens, total symbol length.

    'symbol length' is the sum over the corpus of len(symbols) * freq, i.e. how
    many symbols the encoder would emit with no merges at all. Watching this
    number fall as merges are added is the whole point of BPE, so we record the
    baseline now.
    """
    n_words = len(corpus)
    total_tokens = sum(corpus.values())
    total_symbols = sum(len(symbols) * freq for symbols, freq in corpus.items())
    return {
        "unique_words": n_words,
        "total_tokens": total_tokens,
        "total_symbols_no_merges": total_symbols,
    }


def load_corpus(path=None):
    """Read a text file (or fall back to the built-in sample) into a corpus."""
    if path is None:
        return build_corpus(SAMPLE_TEXT)
    with open(path, "r", encoding="utf-8") as f:
        return build_corpus(f.read())


if __name__ == "__main__":
    corpus = load_corpus()

    print("initial per-word symbol sequences:")
    for symbols, freq in sorted(corpus.items(), key=lambda kv: -kv[1]):
        print(f"  {freq:3d} x {' '.join(symbols)}")

    vocab = initial_vocabulary(corpus)
    print(f"\ninitial vocabulary ({len(vocab)} symbols): {vocab}")

    stats = corpus_stats(corpus)
    print("\ncorpus stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
