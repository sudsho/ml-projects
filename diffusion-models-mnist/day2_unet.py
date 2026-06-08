"""Day 2 - the U-Net noise predictor for the DDPM.

The reverse process needs a network eps_theta(x_t, t) that, given a noised image
and its timestep, predicts the noise that was added. We use a small U-Net: an
encoder that downsamples while growing channels, a bottleneck, and a decoder
that upsamples back to 28x28 while concatenating the matching encoder feature
maps (the skip connections that give a U-Net its shape).

Two details matter for diffusion specifically:

  * The timestep t is shared across the whole image, so we embed it with a
    sinusoidal positional encoding (same idea as in transformers) and inject the
    embedding into every residual block via a small MLP.
  * Residual blocks with GroupNorm + SiLU keep training stable at this depth.

Day 3 wires this into the training loop on the simplified DDPM loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim):
    """Map an integer timestep to a `dim`-vector of sines and cosines.

    Identical in spirit to the transformer positional encoding: low frequencies
    capture coarse position, high frequencies capture fine detail, and the
    network can linearly combine them to recover t."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # zero-pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """Two 3x3 convs with GroupNorm/SiLU and an additive timestep embedding.

    The time embedding is projected to the block's channel count and added after
    the first conv, broadcasting over the spatial dimensions. A 1x1 conv on the
    skip path matches channel counts when in_ch != out_ch."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small three-level U-Net mapping (x_t, t) -> predicted noise.

    Channels grow 1 -> 64 -> 128 -> 256 on the way down. Downsampling is a
    strided conv, upsampling is nearest-neighbour followed by a conv. Skip
    connections concatenate encoder features into the decoder at each level."""

    def __init__(self, base_ch=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

        # encoder
        self.down1 = ResidualBlock(base_ch, base_ch, time_dim)
        self.down2 = ResidualBlock(base_ch, base_ch * 2, time_dim)
        self.down3 = ResidualBlock(base_ch * 2, base_ch * 4, time_dim)
        self.downsample = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        # bottleneck
        self.mid = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)

        # decoder (channels double because of the concatenated skips)
        self.up2 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2, time_dim)
        self.up1 = ResidualBlock(base_ch * 2 + base_ch, base_ch, time_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))

        x = self.in_conv(x)            # 28x28, base_ch
        h1 = self.down1(x, t_emb)      # 28x28
        h = self.downsample(h1)        # 14x14
        h2 = self.down2(h, t_emb)      # 14x14
        h = self.downsample2(h2)       # 7x7
        h3 = self.down3(h, t_emb)      # 7x7

        h = self.mid(h3, t_emb)

        h = F.interpolate(h, scale_factor=2, mode="nearest")   # 14x14
        h = self.up2(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")   # 28x28
        h = self.up1(torch.cat([h, h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


if __name__ == "__main__":
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params/1e6:.2f}M")

    # shape sanity check: a batch of noised images and their timesteps
    x = torch.randn(8, 1, 28, 28)
    t = torch.randint(0, 300, (8,))
    eps = model(x, t)
    assert eps.shape == x.shape, eps.shape
    print(f"input {tuple(x.shape)} -> predicted noise {tuple(eps.shape)}")

    # the timestep embedding should differ across timesteps but be deterministic
    e = sinusoidal_embedding(torch.tensor([0, 0, 150]), 256)
    print(f"emb[t=0] == emb[t=0]: {torch.allclose(e[0], e[1])}")
    print(f"emb[t=0] == emb[t=150]: {torch.allclose(e[0], e[2])}")
