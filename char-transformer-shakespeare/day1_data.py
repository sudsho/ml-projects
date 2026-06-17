"""Day 1 - data pipeline for a character-level transformer on tiny-shakespeare.

Before any modeling we need three things:

  1. a character vocabulary with string <-> integer mappings,
  2. a train/val split of the encoded corpus, and
  3. a way to draw random fixed-length context windows in batches.

A decoder-only language model is trained to predict the next character at every
position, so a single training example is a context window x = tokens[i:i+T] and
its target y = tokens[i+1:i+T+1] (the same window shifted by one). Drawing random
starting offsets each step is the usual cheap substitute for shuffling, since the
corpus is small enough to keep entirely in memory.

Days 2-4 add the attention, the full transformer block, and the training loop.
"""

import os
import urllib.request

import torch

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DATA_PATH = os.path.join(os.path.dirname(__file__), "input.txt")


def load_corpus(path=DATA_PATH):
    """Return the raw text, downloading the tiny-shakespeare corpus on first use."""
    if not os.path.exists(path):
        print(f"downloading tiny-shakespeare to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharVocab:
    """Maps the unique characters of a corpus to a contiguous integer range.

    Character-level modeling keeps the vocabulary tiny (~65 symbols for
    Shakespeare), which sidesteps subword tokenization entirely - every distinct
    character simply gets its own id, assigned in sorted order for determinism.
    """

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def size(self):
        """Number of distinct characters - the model's vocabulary size."""
        return len(self.chars)

    def encode(self, s):
        """Turn a string into a 1-D LongTensor of character ids."""
        return torch.tensor([self.stoi[ch] for ch in s], dtype=torch.long)

    def decode(self, ids):
        """Turn an iterable of ids back into a string."""
        return "".join(self.itos[int(i)] for i in ids)


def train_val_split(data, val_frac=0.1):
    """Split the encoded corpus into train/val by a contiguous tail slice.

    For a single long document we hold out the final `val_frac` of characters
    rather than shuffling, so the validation text never leaks into training.
    """
    n_val = int(len(data) * val_frac)
    n_train = len(data) - n_val
    return data[:n_train], data[n_train:]


class CharDataset:
    """Serves random (context, target) windows from an encoded split.

    Each call to `get_batch` picks `batch_size` random offsets and slices a
    block of `block_size` tokens starting at each, with the target being the
    same block shifted right by one position.
    """

    def __init__(self, data, block_size=128, device="cpu"):
        if len(data) <= block_size:
            raise ValueError("corpus shorter than one context window")
        self.data = data
        self.block_size = block_size
        self.device = device

    def __len__(self):
        # number of distinct starting positions for a full window
        return len(self.data) - self.block_size

    def get_batch(self, batch_size, generator=None):
        """Return (x, y) tensors of shape (batch_size, block_size)."""
        high = len(self.data) - self.block_size
        ix = torch.randint(high, (batch_size,), generator=generator)
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)


def build_pipeline(block_size=128, device="cpu"):
    """Wire the full day-1 pipeline together and return the pieces days 2-4 use.

    Batch size is not fixed here - it is supplied per call to `get_batch`, so the
    pipeline only needs the context window length and the target device.
    """
    text = load_corpus()
    vocab = CharVocab(text)
    encoded = vocab.encode(text)
    train_data, val_data = train_val_split(encoded)
    train_ds = CharDataset(train_data, block_size, device)
    val_ds = CharDataset(val_data, block_size, device)
    return vocab, train_ds, val_ds


if __name__ == "__main__":
    torch.manual_seed(1337)
    vocab, train_ds, val_ds = build_pipeline(block_size=128)

    print(f"vocab size      : {vocab.size}")
    print(f"train tokens    : {len(train_ds.data):,}")
    print(f"val tokens      : {len(val_ds.data):,}")
    print(f"train windows   : {len(train_ds):,}")

    gen = torch.Generator().manual_seed(0)
    xb, yb = train_ds.get_batch(batch_size=4, generator=gen)
    print(f"x batch shape   : {tuple(xb.shape)}")
    print(f"y batch shape   : {tuple(yb.shape)}")

    # sanity check: y is x shifted by one, so y[:, :-1] == x[:, 1:]
    assert torch.equal(xb[:, 1:], yb[:, :-1]), "target must be context shifted by 1"

    sample = vocab.decode(xb[0][:60])
    print(f"decoded context : {sample!r}")
