"""Day 3 - the full Vision Transformer and its training loop.

Days 1 and 2 built the parts in isolation: PatchEmbedding turns an image into a
sequence of CLS + patch tokens, and EncoderBlock lets those tokens exchange
information through pre-norm attention and an MLP. Day 3 stacks the blocks into a
complete classifier and writes the loop that actually trains it on CIFAR-10.

The assembly is deliberately plain:

    image
      -> PatchEmbedding        (B, C, H, W) -> (B, N+1, D)   [CLS prepended]
      -> depth x EncoderBlock  (B, N+1, D) -> (B, N+1, D)
      -> LayerNorm
      -> take token 0 (CLS)    (B, D)
      -> Linear head           (B, num_classes)

Only the CLS token feeds the classifier. It carries no patch content of its own;
it starts as a learned constant and, through attention, aggregates whatever the
patches expose. Reading the head off CLS rather than mean-pooling the patches is
the original ViT choice and keeps a single, inspectable "summary" vector that day
4 will visualize.

Training choices that matter on a small dataset like CIFAR-10:

  - AdamW. Transformers have no convolutional inductive bias, so they lean on the
    optimizer. AdamW decouples weight decay from the gradient step, which is the
    setting ViT was tuned for.
  - Cosine LR schedule with a short linear warmup. Attention logits are unstable
    in the first few hundred steps; warmup avoids an early divergence, and the
    cosine decay anneals smoothly to near zero by the final epoch.
  - Label smoothing. Spreading a little probability mass off the true class stops
    the model from driving logits to extremes and is a cheap, reliable regularizer
    for a from-scratch ViT that would otherwise overfit 50k images.

Day 4 loads a trained model, evaluates the test set, and visualizes the CLS
attention maps.
"""

import math

import torch
import torch.nn as nn

from day1_patch_embed import (
    CHANNELS,
    EMBED_DIM,
    IMAGE_SIZE,
    PATCH_SIZE,
    PatchEmbedding,
    get_dataloaders,
)
from day2_attention import EncoderBlock

NUM_CLASSES = 10


class VisionTransformer(nn.Module):
    """ViT-Tiny-ish classifier sized for CIFAR-10.

    The defaults (embed_dim 192, 3 heads, depth 6) are small enough to train on a
    single GPU in minutes per epoch while staying faithful to the architecture.
    """

    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 channels=CHANNELS, embed_dim=EMBED_DIM, depth=6, num_heads=3,
                 mlp_ratio=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, channels, embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        # A plain stack of identical pre-norm blocks - depth is the only knob.
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final norm before the head; standard for pre-norm transformers, whose
        # residual stream is otherwise never normalized on the way out.
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Truncated-normal Linear weights, zero bias - the reference ViT init.
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        x = self.patch_embed(x)          # (B, N+1, D), CLS at index 0
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]                    # (B, D) - the classification summary token
        return self.head(cls)


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Linear warmup then cosine decay, returned as a LambdaLR multiplier.

    The lambda returns a factor in [0, 1] applied to the base LR: it ramps up
    linearly for `warmup_steps`, then follows half a cosine down to
    `min_lr_ratio` by `total_steps`.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Single pass over the training set; steps the LR schedule every batch."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        # Clip before stepping - attention models occasionally spike a gradient and
        # a single bad step can undo an epoch of progress.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def train(epochs=2, batch_size=128, lr=3e-4, weight_decay=0.05,
          label_smoothing=0.1, warmup_frac=0.1, device=None):
    """Wire the model, data, optimizer and schedule together and train.

    Defaults to a short 2-epoch smoke run so this file is runnable end to end; the
    real day-4 evaluation bumps epochs up. Returns the trained model so a caller
    can checkpoint or evaluate it.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_dataloaders(batch_size=batch_size)

    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_frac * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device: {device} | trainable params: {n_params/1e6:.2f}M | "
          f"steps: {total_steps} (warmup {warmup_steps})")

    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, device)
        last_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:2d}/{epochs} | loss {loss:.4f} | "
              f"train acc {acc:.4f} | lr {last_lr:.2e}")

    return model


if __name__ == "__main__":
    # Sanity check the forward path on a dummy batch without touching the dataset,
    # then leave the real training behind the explicit call below.
    model = VisionTransformer()
    dummy = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    logits = model(dummy)
    print("logits:", tuple(logits.shape), "expected", (4, NUM_CLASSES))
    assert logits.shape == (4, NUM_CLASSES)

    # Schedule shape check: warmup rises, cosine tail falls below the peak.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, warmup_steps=10, total_steps=100)
    factors = []
    for _ in range(100):
        opt.step()
        factors.append(sched.get_last_lr()[0])
        sched.step()
    print(f"lr peak {max(factors):.2e} at step {factors.index(max(factors))}, "
          f"final {factors[-1]:.2e}")
    assert factors[-1] < max(factors)
    print("ok - run train() to fit on CIFAR-10")
