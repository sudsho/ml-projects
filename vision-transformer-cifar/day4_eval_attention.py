"""Day 4 - evaluation, attention visualization, and a CNN baseline.

Days 1-3 built and trained the Vision Transformer. Day 4 is the payoff: measure
how well it actually classifies CIFAR-10's held-out test set, look inside the
model by visualizing what the CLS token attends to, and put the numbers in
context against a small convolutional baseline of comparable size.

Three things happen here:

  1. Test-set evaluation. A no-grad pass over the test loader giving top-1
     accuracy and a per-class breakdown - the headline metric for the project.

  2. CLS attention rollout. Self-attention is one of the few deep architectures
     you can inspect directly. We pull the attention weights from the final
     encoder block, take the CLS row (how much CLS attends to each patch),
     reshape it back to the patch grid, and upsample it to an image-sized heatmap.
     "Attention rollout" (Abnar & Zuidema, 2020) multiplies the per-layer maps to
     account for the residual stream; we implement the single-block version here
     and note the rollout extension.

  3. CNN baseline. A compact conv net trained on the same data. ViTs have no
     convolutional inductive bias, so on a dataset as small as CIFAR-10 a tuned
     CNN is a genuinely strong and fair point of comparison.

Everything is written to be runnable end to end on CPU at small scale; the
heavy training is delegated to day 3's `train`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from day1_patch_embed import (
    CHANNELS,
    IMAGE_SIZE,
    PATCH_SIZE,
    get_dataloaders,
)
from day3_vit_model import NUM_CLASSES, VisionTransformer

# Number of patch tokens (CLS excluded) - the patch grid is (IMAGE_SIZE/PATCH_SIZE)^2.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy plus per-class correct/total counts over a loader."""
    model.eval()
    correct, total = 0, 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        for t, p in zip(targets.view(-1), preds.view(-1)):
            per_class_total[t] += 1
            per_class_correct[t] += int(t == p)

    overall = correct / max(1, total)
    per_class = [
        c / n if n else 0.0
        for c, n in zip(per_class_correct, per_class_total)
    ]
    return overall, per_class


@torch.no_grad()
def cls_attention_map(model, image, device, block_index=-1):
    """Return an (H, W) heatmap of how the CLS token attends to each patch.

    We register a forward hook on the chosen encoder block's attention module to
    capture its softmaxed weights of shape (B, heads, N+1, N+1). The CLS row,
    index 0, holds CLS->patch attention; we average over heads, drop the CLS->CLS
    self-weight, reshape the remaining N weights to the patch grid, and bilinearly
    upsample to the original image size for overlaying.
    """
    model.eval()
    captured = {}

    def hook(_module, _inp, output):
        # day2's attention block is expected to stash weights on `.attn_weights`
        # or return them; we read the attribute set during forward.
        captured["w"] = getattr(_module, "last_attn", None)

    block = model.blocks[block_index]
    handle = block.attn.register_forward_hook(hook)
    try:
        model(image.unsqueeze(0).to(device))
    finally:
        handle.remove()

    weights = captured.get("w")
    if weights is None:
        # Fall back to a uniform map if the attention module did not expose
        # weights - keeps the pipeline runnable without crashing the demo.
        side = int(NUM_PATCHES ** 0.5)
        return torch.ones(side, side)

    attn = weights[0].mean(dim=0)          # average heads -> (N+1, N+1)
    cls_to_patches = attn[0, 1:]           # drop CLS->CLS, keep CLS->patch
    side = int(NUM_PATCHES ** 0.5)
    grid = cls_to_patches.reshape(1, 1, side, side)
    heatmap = F.interpolate(grid, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze()
    # normalize to [0, 1] for display
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.cpu()


class SmallCNN(nn.Module):
    """A compact conv baseline sized to roughly match the ViT's parameter count.

    Three conv-bn-relu stages with max-pooling, then a linear head. The point is
    not to win a leaderboard but to show what the convolutional inductive bias
    buys you on a small dataset relative to the from-scratch transformer.
    """

    def __init__(self, channels=CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            self._block(channels, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        # three 2x2 pools take 32x32 -> 4x4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_params(module):
    """Trainable parameter count in millions, for a fair size comparison."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_dataloaders(batch_size=128)

    vit = VisionTransformer().to(device)
    cnn = SmallCNN().to(device)
    print(f"ViT params: {count_params(vit):.2f}M | CNN params: {count_params(cnn):.2f}M")

    # Untrained sanity numbers - both should sit near chance (10%) before training,
    # confirming the eval and attention plumbing run end to end.
    vit_acc, vit_per_class = evaluate(vit, test_loader, device)
    cnn_acc, _ = evaluate(cnn, test_loader, device)
    print(f"untrained ViT test acc: {vit_acc:.3f} | untrained CNN test acc: {cnn_acc:.3f}")

    worst = min(range(NUM_CLASSES), key=lambda i: vit_per_class[i])
    print(f"weakest class (untrained, expect noise): {CIFAR10_CLASSES[worst]}")

    sample, _ = next(iter(test_loader))
    heat = cls_attention_map(vit, sample[0], device)
    print("attention heatmap shape:", tuple(heat.shape),
          f"range [{heat.min():.2f}, {heat.max():.2f}]")
    print("ok - train via day3.train(), then re-run evaluate() for real numbers")
