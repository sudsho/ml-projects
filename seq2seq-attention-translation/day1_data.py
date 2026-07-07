"""Day 1 - data pipeline for a seq2seq translation model with attention.

Before any encoder/decoder/attention code we need the boring-but-critical
plumbing that every neural MT model sits on:

  1. tokenization of parallel sentence pairs (source language -> target language),
  2. a Vocabulary per side that maps tokens <-> integer ids and reserves the
     special symbols <pad>, <sos>, <eos>, <unk>, and
  3. batching that pads variable-length sentences to a common length and reports
     the true lengths so later code can pack/mask the padding out.

The attention decoder (day 2-3) will consume batches of shape (batch, time) plus
the source lengths; getting those shapes and the <sos>/<eos> conventions right
now saves a lot of debugging once gradients are flowing.

Days 2-4 add the bidirectional-LSTM encoder + Bahdanau attention, the
teacher-forced training loop, and beam-search decoding with BLEU.
"""

import re
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]

# A tiny hand-written English<->French corpus so the file runs standalone; swap
# in a real parallel corpus (e.g. Multi30k / Tatoeba) by editing load_pairs().
_TOY_PAIRS = [
    ("i am cold", "j ai froid"),
    ("she is happy", "elle est heureuse"),
    ("we are tired", "nous sommes fatigues"),
    ("he is a student", "il est etudiant"),
    ("they are friends", "ils sont amis"),
    ("the cat is black", "le chat est noir"),
    ("i love this book", "j aime ce livre"),
    ("it is very cold today", "il fait tres froid aujourd hui"),
]

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(sentence):
    """Lowercase and split a sentence into word tokens.

    Deliberately minimal - lowercase then grab maximal runs of letters. A real
    pipeline would use a subword tokenizer, but word-level keeps day 1 legible
    and the vocabulary small.
    """
    return _TOKEN_RE.findall(sentence.lower())


def load_pairs():
    """Return a list of (source_tokens, target_tokens) tuples.

    Replace the toy list with file reading for a real corpus; the rest of the
    pipeline is agnostic to where the pairs come from.
    """
    return [(tokenize(src), tokenize(tgt)) for src, tgt in _TOY_PAIRS]


class Vocabulary:
    """Maps tokens to contiguous ids, reserving the four special symbols first.

    Ids 0..3 are always <pad>, <sos>, <eos>, <unk> so that, for example, the
    padding id is a stable 0 that the loss can ignore. Tokens rarer than
    `min_freq` collapse to <unk>, which keeps the embedding table small and gives
    the model a way to represent words it never saw at training time.
    """

    def __init__(self, sentences, min_freq=1):
        counts = Counter(tok for sent in sentences for tok in sent)
        self.itos = list(SPECIALS)
        for tok, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    def encode(self, tokens, add_bos_eos=True):
        """Turn a token list into a 1-D LongTensor of ids.

        When `add_bos_eos` is set the sequence is wrapped in <sos> ... <eos>,
        which is what the decoder side needs; the encoder side usually skips it.
        Unknown tokens map to <unk> rather than raising.
        """
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens]
        if add_bos_eos:
            ids = [self.stoi[SOS]] + ids + [self.stoi[EOS]]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        """Turn ids back into tokens, stopping at <eos> and dropping specials."""
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == EOS:
                break
            if tok not in (PAD, SOS):
                out.append(tok)
        return out


class TranslationDataset(Dataset):
    """Yields (source_ids, target_ids) pairs, both already numericalized.

    The source is encoded without <sos>/<eos>; the target carries them so the
    decoder has an explicit start token to condition on and a stop token to learn.
    """

    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]
        src = self.src_vocab.encode(src_tokens, add_bos_eos=False)
        tgt = self.tgt_vocab.encode(tgt_tokens, add_bos_eos=True)
        return src, tgt


def make_collate_fn(src_pad, tgt_pad):
    """Build a collate_fn that pads a batch and returns the real source lengths.

    Source lengths are handed back so day 2's encoder can pack the sequence and
    the attention mechanism can mask padded positions out of the softmax - if we
    forget this the model happily attends to <pad> and the alignments turn to mush.
    """

    def collate(batch):
        """Pad one batch of (src, tgt) pairs and return the true source lengths."""
        srcs, tgts = zip(*batch)
        src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
        src_pad_batch = pad_sequence(srcs, batch_first=True, padding_value=src_pad)
        tgt_pad_batch = pad_sequence(tgts, batch_first=True, padding_value=tgt_pad)
        return src_pad_batch, src_lens, tgt_pad_batch

    return collate


def build_dataloaders(batch_size=4):
    """Assemble vocabularies, dataset, and a padded DataLoader in one call."""
    pairs = load_pairs()
    src_vocab = Vocabulary([src for src, _ in pairs])
    tgt_vocab = Vocabulary([tgt for _, tgt in pairs])
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    collate = make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    return loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=4)
    print(f"source vocab: {len(src_vocab)} tokens, target vocab: {len(tgt_vocab)} tokens")

    src_batch, src_lens, tgt_batch = next(iter(loader))
    print(f"src batch shape {tuple(src_batch.shape)}, lengths {src_lens.tolist()}")
    print(f"tgt batch shape {tuple(tgt_batch.shape)}")

    # sanity check: decode the first target row back to text
    print("decoded target[0]:", " ".join(tgt_vocab.decode(tgt_batch[0])))
    assert src_batch.shape[0] == tgt_batch.shape[0]
    assert src_lens.max().item() == src_batch.shape[1]
    print("data pipeline ok")
