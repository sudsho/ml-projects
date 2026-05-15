"""
Day 4 of neural style transfer.

Multi-style blending experiments and a comparison grid. Instead of a single
style reference we accept N styles with weights summing to 1, average their
Gram matrices weighted accordingly, and run the same L-BFGS optimization.

Also produces a single side-by-side comparison image (content + each pure
style result + each blend) using PIL so the README can show one figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from day2_losses import StyleTransferObjective, gram_matrix
from day3_optimize import OptimConfig, run_lbfgs


@dataclass
class BlendSpec:
    style_paths: List[Path]
    weights: List[float]

    def __post_init__(self) -> None:
        if len(self.style_paths) != len(self.weights):
            raise ValueError("style_paths and weights must have the same length")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            self.weights = [w / total for w in self.weights]


def blended_style_targets(
    extractor: VGG19FeatureExtractor,
    spec: BlendSpec,
    device: torch.device,
    image_size: int,
) -> Dict[str, torch.Tensor]:
    """Weighted average of Gram matrices across multiple style images."""
    accumulated: Dict[str, torch.Tensor] = {}
    for path, w in zip(spec.style_paths, spec.weights):
        img = load_image(path, image_size).to(device)
        feats = extractor(img)
        for layer in STYLE_LAYERS:
            g = gram_matrix(feats[layer])
            if layer not in accumulated:
                accumulated[layer] = w * g
            else:
                accumulated[layer] = accumulated[layer] + w * g
    return accumulated


def stylize(
    content_path: Path,
    spec: BlendSpec,
    out_path: Path,
    device: torch.device,
    image_size: int = 384,
    cfg: OptimConfig = OptimConfig(),
) -> Path:
    extractor = VGG19FeatureExtractor().to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False

    content = load_image(content_path, image_size).to(device)
    content_feats = extractor(content)
    content_targets = {l: content_feats[l].detach() for l in CONTENT_LAYERS}
    style_targets = blended_style_targets(extractor, spec, device, image_size)

    objective = StyleTransferObjective(
        content_targets=content_targets,
        style_gram_targets=style_targets,
    )
    image = content.clone().requires_grad_(True)
    final = run_lbfgs(image, extractor, objective, cfg)
    save_image(final.detach(), out_path)
    return out_path


def make_comparison_grid(tiles: List[Tuple[str, Path]], out_path: Path,
                         tile_size: int = 256, padding: int = 8) -> Path:
    """Build a horizontal strip with labels written above each tile."""
    from PIL import ImageDraw, ImageFont
    n = len(tiles)
    label_h = 24
    w = n * tile_size + (n + 1) * padding
    h = tile_size + label_h + 2 * padding
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, (label, path) in enumerate(tiles):
        x = padding + i * (tile_size + padding)
        img = Image.open(path).convert("RGB").resize((tile_size, tile_size))
        canvas.paste(img, (x, label_h + padding))
        draw.text((x + 4, 4), label, fill=(20, 20, 20), font=font)
    canvas.save(out_path)
    return out_path


def parse_blend_arg(s: str) -> BlendSpec:
    """Format: 'starry.jpg:0.5,wave.jpg:0.5'."""
    paths, weights = [], []
    for chunk in s.split(","):
        path, w = chunk.split(":")
        paths.append(Path(path.strip()))
        weights.append(float(w))
    return BlendSpec(paths, weights)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-style blending experiments")
    ap.add_argument("--content", type=Path, required=True)
    ap.add_argument("--blends", type=parse_blend_arg, nargs="+", required=True,
                    help="One or more blend specs like 'a.jpg:0.7,b.jpg:0.3'")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OptimConfig(num_steps=args.steps)

    tiles: List[Tuple[str, Path]] = [("content", args.content)]
    for i, spec in enumerate(args.blends):
        out = args.out_dir / f"blend_{i:02d}.png"
        stylize(args.content, spec, out, device, args.image_size, cfg)
        label = " + ".join(f"{p.stem}({w:.2f})" for p, w in zip(spec.style_paths, spec.weights))
        tiles.append((label, out))

    grid_path = args.out_dir / "comparison_grid.png"
    make_comparison_grid(tiles, grid_path)
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
