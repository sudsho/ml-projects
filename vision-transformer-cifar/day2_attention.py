"""Day 2 - self-attention and the pre-norm transformer encoder block.

Day 1 turned a CIFAR-10 image into a sequence of patch tokens (CLS + 64 patches,
each a 192-dim vector). Day 2 builds the machinery that lets those tokens talk to
each other: scaled dot-product attention, the multi-head wrapper around it, and
the encoder block that stacks attention and an MLP with residual connections.

A ViT encoder is the *same* transformer used for language, minus the causal mask.
Image patches are not ordered the way text is - patch 7 is no more "in the future"
than patch 3 - so every token attends to every other token, CLS included. That is
the one structural difference from the decoder-only model in this repo's
char-transformer project, which masks out future positions.

Three pieces live here:

  - Scaled dot-product attention. For queries Q, keys K, values V, compute
    softmax(QK^T / sqrt(d_k)) V. The 1/sqrt(d_k) scale keeps the dot products from
    growing with head dimension and pushing softmax into a near-one-hot regime
    where gradients vanish.
  - Multi-head self-attention. Project the input into h independent (Q,K,V) sets,
    run attention in each head in parallel, concatenate, and project back. Heads
    let the model attend to several relationship types at once (e.g. one head on
    nearby patches, another on the CLS token).
  - Pre-norm encoder block. LayerNorm *before* each sub-layer rather than after.
    Pre-norm keeps a clean residual highway from input to output, which makes deep
    transformers train stably without the learning-rate warmup gymnastics that
    post-norm needs. Block = x + Attn(LN(x)); then x + MLP(LN(x)).

Day 3 stacks these blocks into the full ViT with a classification head and the
training loop.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from day1_patch_embed import EMBED_DIM


def scaled_dot_product_attention(q, k, v, dropout=None):
    """Core attention: softmax(QK^T / sqrt(d_k)) V.

    Shapes (B = batch, H = heads, N = sequence length, d = head dim):
        q, k, v : (B, H, N, d)
        returns : (B, H, N, d) context, plus the (B, H, N, N) attention weights so
                  day 4 can visualize what the CLS token looks at.

    There is no mask: a ViT encoder is bidirectional, every token sees every other.
    """
    d_k = q.size(-1)
    # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N) similarity of every query/key pair
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    context = torch.matmul(attn, v)   # (B, H, N, d)
    return context, attn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over a token sequence.

    A single fused Linear produces Q, K and V for all heads at once (3*embed_dim
    out), which is cheaper than three separate projections and is the standard
    implementation trick. We reshape to (B, H, N, d), attend per head, then merge
    the heads back and apply the output projection.
    """

    def __init__(self, embed_dim=EMBED_DIM, num_heads=3, attn_dropout=0.0,
                 proj_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Keep the most recent attention map around for visualization on day 4.
        self.last_attn = None

    def forward(self, x):
        # x: (B, N, D)
        b, n, d = x.shape

        # (B, N, 3D) -> (3, B, H, N, d) so we can unpack q, k, v cleanly.
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each (B, H, N, d)

        context, attn = scaled_dot_product_attention(q, k, v, self.attn_dropout)
        self.last_attn = attn.detach()

        # Merge heads: (B, H, N, d) -> (B, N, H*d = D)
        context = context.transpose(1, 2).reshape(b, n, d)
        out = self.proj_dropout(self.proj(context))
        return out


class MLP(nn.Module):
    """Position-wise feed-forward network: D -> hidden -> D with GELU.

    Applied identically to every token. The hidden width is usually a multiple of
    the model width (mlp_ratio=4 is the ViT default) and is where most of the
    block's parameters and per-token "thinking" capacity live.
    """

    def __init__(self, embed_dim=EMBED_DIM, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class EncoderBlock(nn.Module):
    """One pre-norm transformer encoder block.

        x = x + MHSA(LN(x))
        x = x + MLP(LN(x))

    The residual adds run on the *un-normalized* stream, so gradients flow back
    through an identity path no matter how deep the stack gets - this is why
    pre-norm transformers are forgiving to train.
    """

    def __init__(self, embed_dim=EMBED_DIM, num_heads=3, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads,
                                           attn_dropout=dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    # Shape checks on a dummy token sequence - no dataset needed. A block must be
    # shape-preserving: (B, N, D) in, (B, N, D) out.
    b, n, d = 8, 65, EMBED_DIM   # 64 patches + 1 CLS = 65 tokens
    tokens = torch.randn(b, n, d)

    block = EncoderBlock()
    out = block(tokens)
    print("input :", tuple(tokens.shape))
    print("output:", tuple(out.shape))
    assert out.shape == tokens.shape

    # Attention weights must be a valid distribution: per-query rows sum to 1.
    attn = block.attn.last_attn        # (B, H, N, N)
    row_sums = attn.sum(dim=-1)
    print("attn map:", tuple(attn.shape), "row-sum ~1:",
          torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print("ok")
