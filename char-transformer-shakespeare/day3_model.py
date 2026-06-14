"""Day 3 - the transformer block and the full decoder-only model.

Day 2 gave us masked multi-head self-attention. Today we wrap it into a proper
transformer block and stack those blocks into the full language model:

  1. a position-wise feed-forward network (the MLP that follows attention),
  2. a pre-norm transformer block with residual connections around both the
     attention and the MLP sub-layers,
  3. the CharTransformer itself - token + positional embeddings, a stack of
     blocks, a final layernorm, and a linear head projecting back to the vocab.

We use the *pre-norm* arrangement (LayerNorm before each sub-layer rather than
after) because it makes deep stacks far easier to train - gradients flow through
the residual path untouched by the norm. Day 4 trains this model and samples
from it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from day2_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise MLP: expand to 4x width, GELU, project back, dropout.

    Applied independently at every position. The 4x expansion ratio is the
    convention from the original transformer - it gives the block capacity to
    transform each token's representation between attention mixes.
    """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One pre-norm transformer block: attention then MLP, each residual.

        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))

    Normalising the *input* of each sub-layer (rather than its output) keeps a
    clean identity path from input to output, which is what lets us stack many
    of these without the training going unstable.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    """Decoder-only transformer over character ids.

    Token ids and their positions are embedded and summed, run through a stack
    of pre-norm blocks, normalised once more, then projected to per-character
    logits. forward() optionally returns the cross-entropy loss when targets are
    supplied, which is what the day-4 training loop consumes.
    """

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # small normal init for linears/embeddings, as in nanoGPT
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )
        return logits, loss

    def num_params(self):
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(1337)

    vocab_size = 65  # tiny-shakespeare character set from day 1
    model = CharTransformer(vocab_size, n_embd=128, n_head=4, n_layer=4,
                            block_size=128)

    B, T = 8, 64
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    logits, loss = model(idx, targets)
    print(f"logits shape : {tuple(logits.shape)}")
    print(f"loss         : {loss.item():.4f}")
    # untrained loss should sit near ln(vocab_size), i.e. a uniform guess
    print(f"ln(vocab)    : {torch.log(torch.tensor(float(vocab_size))).item():.4f}")
    print(f"param count  : {model.num_params():,}")
