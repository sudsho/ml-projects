"""
Day 1 of the RealNVP normalizing flow on MNIST.

A normalizing flow models a density by pushing a simple base distribution (a
standard Gaussian) through an *invertible* map f. The change-of-variables
formula then gives an exact log-likelihood:

    log p_x(x) = log p_z(f(x)) + log |det df/dx|

so the whole game is to build an f that (a) is expressive, (b) is cheap to
invert, and (c) has a tractable Jacobian determinant. RealNVP's answer is the
affine coupling layer, which this file builds along with the input pipeline that
makes MNIST usable by a continuous-density model.

Two preprocessing steps have to happen before any coupling layer sees the data:

  1. Dequantization. MNIST pixels are discrete {0,...,255}. A continuous density
     can put arbitrarily high likelihood on those exact points (spikes of
     infinite density), which is degenerate. Adding uniform noise in [0,1) to
     each pixel spreads each integer over its bin, turning the discrete pmf into
     a continuous density whose likelihood lower-bounds the true one.

  2. Logit transform. After scaling to [0,1] the data lives on a bounded cube,
     but a Gaussian base lives on all of R^d. We map (0,1) -> R with a logit,
     squeezing away from the boundaries first (the RealNVP alpha trick) so the
     logit never blows up. This is itself an invertible transform, so its
     log-determinant has to be tracked and added to the running total.

The coupling layer then splits the vector into two halves via a binary mask:
one half passes through untouched, and its values condition a small network that
outputs a per-dimension scale (log s) and translation (t) applied to the other
half. Because the transformed half never feeds back into the network that
produced its own parameters, the Jacobian is triangular and its log-determinant
is simply the sum of log s over the transformed coordinates - and inverting the
layer is just (y - t) * exp(-log s), no network inversion required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dequantize(x, n_bits=8):
    """Add uniform noise and scale integer pixels into (0, 1).

    x is expected in [0, 1] as returned by ToTensor (i.e. already divided by
    255). We undo that to get back to {0,...,255}, add U[0,1) noise, and divide
    by 256 so the result lands in [0, 1). The +noise is what makes the density
    continuous; without it the model would chase delta spikes at the integers.
    """
    n_vals = 2 ** n_bits
    x = x * (n_vals - 1)            # back to 0..255 (approx, from ToTensor)
    x = x + torch.rand_like(x)      # uniform dequantization noise
    x = x / n_vals                  # into [0, 1)
    return x


def logit_transform(x, alpha=1e-6):
    """Map (0,1) data to R via a squeezed logit; return (y, log_det).

    The squeeze x -> alpha + (1 - 2*alpha)*x keeps values off the 0/1 boundaries
    so logit stays finite. log_det accumulates the per-pixel derivative of the
    whole transform, summed over the feature dimension, because the flow's
    likelihood needs the Jacobian of every step including preprocessing.
    """
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log1p(-s)          # logit(s)
    # d/dx logit(alpha + (1-2a)x) = (1-2a) / (s * (1 - s))
    log_det = (torch.log(torch.tensor(1 - 2 * alpha))
               - torch.log(s) - torch.log1p(-s))
    log_det = log_det.flatten(1).sum(dim=1)
    return y, log_det


class ScaleTranslateNet(nn.Module):
    """Small MLP producing (log_scale, translate) for a coupling layer.

    Takes the full (masked) input so shapes stay uniform across layers, and emits
    two vectors of the input dimension. log_scale is passed through tanh and a
    learned per-layer factor, a standard RealNVP stabilizer that keeps the scale
    from exploding early in training.
    """

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * 2),
        )
        self.scale_factor = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h = self.net(x)
        log_scale, translate = h.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale) * self.scale_factor
        return log_scale, translate


class AffineCoupling(nn.Module):
    """One RealNVP affine coupling layer with a fixed binary mask.

    The masked coordinates (mask == 1) are copied through and condition the
    scale/translate network; the complementary coordinates are affinely
    transformed. Forward returns the transformed vector and this layer's
    contribution to the log-determinant (sum of log_scale over the *changed*
    coordinates only). inverse undoes it exactly.
    """

    def __init__(self, dim, mask, hidden=256):
        super().__init__()
        # register the mask as a buffer so .to(device)/state_dict carry it along
        self.register_buffer("mask", mask.float())
        self.st = ScaleTranslateNet(dim, hidden)

    def forward(self, x):
        x_masked = x * self.mask
        log_scale, translate = self.st(x_masked)
        # only the unmasked half is actually transformed
        log_scale = log_scale * (1 - self.mask)
        translate = translate * (1 - self.mask)
        y = x_masked + (1 - self.mask) * (x * torch.exp(log_scale) + translate)
        log_det = log_scale.sum(dim=1)
        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        log_scale, translate = self.st(y_masked)
        log_scale = log_scale * (1 - self.mask)
        translate = translate * (1 - self.mask)
        x = y_masked + (1 - self.mask) * (y - translate) * torch.exp(-log_scale)
        return x


def checkerboard_mask(dim, parity=0):
    """Alternating 0/1 mask over a flattened vector."""
    idx = torch.arange(dim)
    return ((idx + parity) % 2)


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 784  # flattened 28x28 MNIST

    # a fake "image batch" in [0,1] standing in for a ToTensor MNIST batch
    fake = torch.rand(16, 1, 28, 28)
    deq = dequantize(fake)
    assert (deq >= 0).all() and (deq < 1).all(), "dequantized data must be in [0,1)"

    flat = deq.flatten(1)
    z, ld_pre = logit_transform(flat)
    print(f"after logit: mean {z.mean():.3f}, std {z.std():.3f}, "
          f"preproc log|det| shape {tuple(ld_pre.shape)}")

    layer = AffineCoupling(dim, checkerboard_mask(dim, parity=0))
    y, log_det = layer(z)
    recon = layer.inverse(y)

    err = (recon - z).abs().max().item()
    print(f"coupling log_det mean: {log_det.mean().item():.4f}")
    print(f"forward/inverse max reconstruction error: {err:.2e}")
    assert err < 1e-4, "coupling layer must be exactly invertible"
    print("day 1 checks passed: dequantization, logit preprocessing, "
          "invertible affine coupling with tractable log-determinant.")
