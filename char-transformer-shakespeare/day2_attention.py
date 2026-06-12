"""Day 2 - self-attention for the character-level transformer.

The data pipeline from day 1 hands us batches of shape (B, T) of character ids.
Today we build the attention machinery that lets each position gather context
from the positions before it:

  1. a single attention head (scaled dot-product attention with a causal mask),
  2. multi-head self-attention that runs several heads in parallel and mixes
     their outputs with an output projection.

The causal mask is what makes this a *decoder* - position t may only attend to
positions <= t, so the model never peeks at the character it is trying to
predict. Days 3-4 stack these into the full transformer block and train it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single masked self-attention head.

    Projects the input into query/key/value spaces, scores every query against
    every key, masks out the future with a lower-triangular mask, softmaxes into
    attention weights, and returns the weighted sum of the values.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is buffer state, not a parameter - it should not get gradients
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # scaled dot-product scores: (B, T, T), scaled by 1/sqrt(d) so the
        # softmax does not saturate as head_size grows
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_size))
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        return weights @ v   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected.

    Running `n_head` independent heads of size `n_embd // n_head` lets each head
    specialise on a different relationship; the output projection then mixes the
    concatenated results back into the model dimension.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


def causal_mask_is_respected(weights, atol=1e-6):
    """Sanity helper: attention weights must be zero in the upper triangle.

    weights has shape (B, T, T); position i may not attend to any j > i, so every
    strictly-upper-triangular entry should be ~0 after the masked softmax.
    """
    T = weights.shape[-1]
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    return torch.all(weights[..., upper].abs() < atol).item()


if __name__ == "__main__":
    torch.manual_seed(1337)

    B, T, n_embd, n_head = 4, 16, 64, 8
    block_size = 32
    x = torch.randn(B, T, n_embd)

    head = Head(n_embd, head_size=16, block_size=block_size)
    out = head(x)
    print(f"single head out : {tuple(out.shape)}")

    mha = MultiHeadAttention(n_embd, n_head, block_size)
    out = mha(x)
    print(f"multi-head out  : {tuple(out.shape)}")
    print(f"param count     : {sum(p.numel() for p in mha.parameters()):,}")

    # verify the mask: re-run a head and inspect its raw attention weights
    k, q = head.key(x), head.query(x)
    scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(head.head_size))
    scores = scores.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    print(f"causal mask ok  : {causal_mask_is_respected(weights)}")
    print(f"row sums ~1     : {torch.allclose(weights.sum(-1), torch.ones(B, T))}")
