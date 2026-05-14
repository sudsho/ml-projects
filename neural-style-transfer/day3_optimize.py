"""
Day 3 of neural style transfer.

The optimization loop. We start from a clone of the content image (works much
better than starting from white noise in practice), wrap it as a leaf tensor
with requires_grad=True, and hand it to L-BFGS. L-BFGS wants a closure that
recomputes the loss and gradients each call, since it does multiple internal
evaluations per step.

There's also a simple learning rate / step schedule and an option to dump
intermediate snapshots every K iterations so we can build a progress GIF
later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
from PIL import Image
from torchvision import transforms

from day1_features import (
    CONTENT_LAYERS,
    STYLE_LAYERS,
    VGG19FeatureExtractor,
    load_image,
    save_image,
)
from day2_losses import StyleTransferObjective


@dataclass
class OptimConfig:
    num_steps: int = 300
    lr: float = 1.0
    snapshot_every: int = 50
    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 1e-3
    log_every: int = 10
    snapshot_dir: Optional[Path] = None
    content_layers: List[str] = field(default_factory=lambda: list(CONTENT_LAYERS))
    style_layers: List[str] = field(default_factory=lambda: list(STYLE_LAYERS))


def _clamp_(image: torch.Tensor) -> None:
    # In-place clamp into the normalized VGG input range. Without this the
    # generated tensor drifts outside the range the network was trained on
    # and the loss surface gets weird around the edges.
    image.data.clamp_(-2.5, 2.5)


def stylize(content_path: Path,
            style_path: Path,
            output_path: Path,
            config: OptimConfig,
            device: Optional[torch.device] = None) -> torch.Tensor:
    """Run L-BFGS style transfer end-to-end and return the final tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = VGG19FeatureExtractor(
        content_layers=config.content_layers,
        style_layers=config.style_layers,
    ).to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    content_image = load_image(content_path).to(device)
    style_image = load_image(style_path, target_size=content_image.shape[-2:]).to(device)

    with torch.no_grad():
        content_target_feats = extractor(content_image)
        style_target_feats = extractor(style_image)

    # Start from the content image - empirically much faster to converge than
    # starting from noise, and tends to preserve recognizable structure.
    generated = content_image.clone().detach().requires_grad_(True)

    objective = StyleTransferObjective(
        content_layers=config.content_layers,
        style_layers=config.style_layers,
        content_weight=config.content_weight,
        style_weight=config.style_weight,
        tv_weight=config.tv_weight,
    ).to(device)

    optimizer = torch.optim.LBFGS([generated], lr=config.lr, max_iter=20,
                                  tolerance_grad=1e-7, history_size=50)

    if config.snapshot_dir is not None:
        config.snapshot_dir.mkdir(parents=True, exist_ok=True)

    step_count = {"i": 0}

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _clamp_(generated)
        gen_feats = extractor(generated)
        losses = objective(generated, gen_feats,
                           content_target_feats, style_target_feats)
        total = losses["total"]
        total.backward()

        step_count["i"] += 1
        i = step_count["i"]
        if i % config.log_every == 0 or i == 1:
            print(
                f"step {i:4d} | total {total.item():.4f} | "
                f"content {losses['content'].item():.4f} | "
                f"style {losses['style'].item():.6f} | "
                f"tv {losses['tv'].item():.4f}"
            )
        if config.snapshot_dir is not None and i % config.snapshot_every == 0:
            snap = config.snapshot_dir / f"snapshot_{i:04d}.png"
            with torch.no_grad():
                save_image(generated.detach(), snap)
        return total

    # L-BFGS calls closure several times per outer step, so we don't multiply
    # the outer loop count by max_iter - we just step until the budget is up.
    outer_steps = max(1, math.ceil(config.num_steps / 20))
    for _ in range(outer_steps):
        optimizer.step(closure)

    _clamp_(generated)
    with torch.no_grad():
        save_image(generated.detach(), output_path)
    return generated.detach()


def _quick_demo():
    """Mini end-to-end run on whatever sample images we have in ./inputs/."""
    here = Path(__file__).parent
    content = here / "inputs" / "content.jpg"
    style = here / "inputs" / "style.jpg"
    out = here / "outputs" / "stylized.png"
    if not content.exists() or not style.exists():
        print("inputs/content.jpg or inputs/style.jpg not found, skipping demo")
        return
    cfg = OptimConfig(
        num_steps=100,
        snapshot_every=20,
        snapshot_dir=here / "outputs" / "snapshots",
    )
    stylize(content, style, out, cfg)


if __name__ == "__main__":
    _quick_demo()
