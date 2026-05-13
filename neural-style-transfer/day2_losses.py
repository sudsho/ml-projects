"""
Day 2 of neural style transfer.

Content loss, style loss with Gram matrices, and a total variation regularizer.
The three are combined into a single weighted objective used by the
optimization loop on day 3.

Notes:
- Content loss is plain MSE between the generated and target activations at
  the chosen content layer.
- Style loss is MSE between Gram matrices of the generated and target style
  activations, summed over the style layers. Each style layer contributes
  with an optional weight (default uniform).
- Total variation loss is a smoothness prior on the generated image. Helps a
  lot with the high-frequency noise you otherwise get from VGG features.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of a (B, C, H, W) feature map.

    Returns a (B, C, C) tensor normalized by the number of spatial positions,
    matching the convention used in the original Gatys et al. formulation.
    """
    b, c, h, w = features.shape
    flat = features.view(b, c, h * w)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram / (c * h * w)


def content_loss(gen_feats: Mapping[str, torch.Tensor],
                 target_feats: Mapping[str, torch.Tensor],
                 layers: Iterable[str]) -> torch.Tensor:
    loss = torch.zeros((), device=next(iter(gen_feats.values())).device)
    for name in layers:
        loss = loss + F.mse_loss(gen_feats[name], target_feats[name])
    return loss


def style_loss(gen_feats: Mapping[str, torch.Tensor],
               target_feats: Mapping[str, torch.Tensor],
               layers: Iterable[str],
               layer_weights: Optional[Mapping[str, float]] = None) -> torch.Tensor:
    layers = list(layers)
    if layer_weights is None:
        layer_weights = {name: 1.0 / len(layers) for name in layers}

    loss = torch.zeros((), device=next(iter(gen_feats.values())).device)
    for name in layers:
        gen_gram = gram_matrix(gen_feats[name])
        target_gram = gram_matrix(target_feats[name])
        loss = loss + layer_weights[name] * F.mse_loss(gen_gram, target_gram)
    return loss


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV: sum of absolute neighbor differences along H and W."""
    if image.dim() != 4:
        raise ValueError("expected (B, C, H, W)")
    dh = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    dw = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    return dh + dw


class StyleTransferObjective(nn.Module):
    """Weighted combination of content, style, and TV losses."""

    def __init__(self,
                 content_layers: Iterable[str],
                 style_layers: Iterable[str],
                 content_weight: float = 1.0,
                 style_weight: float = 1e6,
                 tv_weight: float = 1e-3,
                 style_layer_weights: Optional[Mapping[str, float]] = None):
        super().__init__()
        self.content_layers = list(content_layers)
        self.style_layers = list(style_layers)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.style_layer_weights = style_layer_weights

    def forward(self,
                generated_image: torch.Tensor,
                gen_feats: Mapping[str, torch.Tensor],
                content_target_feats: Mapping[str, torch.Tensor],
                style_target_feats: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = content_loss(gen_feats, content_target_feats, self.content_layers)
        s = style_loss(gen_feats, style_target_feats, self.style_layers,
                       self.style_layer_weights)
        tv = total_variation_loss(generated_image)
        total = self.content_weight * c + self.style_weight * s + self.tv_weight * tv
        return {"content": c, "style": s, "tv": tv, "total": total}


def _smoke_test():
    """Quick check that everything wires up on random tensors."""
    torch.manual_seed(0)
    content_layers = ["conv4_2"]
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    shapes = {
        "conv1_1": (1, 64, 256, 256),
        "conv2_1": (1, 128, 128, 128),
        "conv3_1": (1, 256, 64, 64),
        "conv4_1": (1, 512, 32, 32),
        "conv4_2": (1, 512, 32, 32),
        "conv5_1": (1, 512, 16, 16),
    }
    gen = {k: torch.randn(*v, requires_grad=True) for k, v in shapes.items()}
    tgt_c = {k: torch.randn(*v) for k, v in shapes.items()}
    tgt_s = {k: torch.randn(*v) for k, v in shapes.items()}
    image = torch.randn(1, 3, 256, 256, requires_grad=True)

    obj = StyleTransferObjective(content_layers, style_layers)
    losses = obj(image, gen, tgt_c, tgt_s)
    losses["total"].backward()
    assert image.grad is not None
    assert all(torch.isfinite(v) for v in losses.values())
    print({k: float(v) for k, v in losses.items()})


if __name__ == "__main__":
    _smoke_test()
