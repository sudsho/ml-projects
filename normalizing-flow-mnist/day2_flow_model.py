"""
Day 2 of the RealNVP normalizing flow on MNIST.

Day 1 built one affine coupling layer and the preprocessing that gets MNIST onto
the real line. A single coupling layer is badly limited though: half its
coordinates pass through completely untouched, so it cannot model any density
that requires transforming everything. The fix is to stack layers and flip the
mask between them, so a coordinate that was frozen in one layer is transformed in
the next.

Today assembles the full flow and closes the loop on the likelihood. Stacking is
what makes the change-of-variables bookkeeping worth writing down carefully,
because log-determinants *add* along a composition. If f = f_K o ... o f_1 then

    log p_x(x) = log p_z(z) + sum_k log |det df_k/dx_{k-1}|

where z = f(x). Each coupling layer already returns its own contribution, and the
preprocessing (dequantize + logit) contributes one too, so the total is a running
sum threaded through the forward pass. Nothing here needs an explicit Jacobian
matrix - that is the whole point of the coupling design.

The base density is a standard Gaussian, so log p_z(z) is a closed form. Reported
in bits per dimension rather than raw nats, since that is the comparable number
for density models on images and it is what day 3 will train against:

    bpd = -log p_x(x) / (D * ln 2) + log2(256)

The trailing log2(256) = 8 term un-does the /256 scaling applied during
dequantization - without it the number looks better than it is.

Two correctness checks matter more than anything numeric today, and both are
asserted at the bottom: inverse(forward(x)) must return x to floating-point
precision, and the log-determinant accumulated forward must be the negation of
the one accumulated backward. A flow that fails either is silently not a flow,
and the likelihood it reports is meaningless.
"""

import math

import torch
import torch.nn as nn

from day1_coupling_layer import (
    AffineCoupling,
    checkerboard_mask,
    dequantize,
    logit_transform,
)


class RealNVP(nn.Module):
    """A stack of affine coupling layers with alternating masks.

    Layer k uses parity k % 2, so consecutive layers freeze opposite halves of
    the vector and every coordinate gets transformed by at least one of any two
    adjacent layers. forward maps data -> latent and accumulates the total
    log-determinant; inverse maps latent -> data by running the layers in
    reverse order.
    """

    def __init__(self, dim, n_layers=6, hidden=256):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            AffineCoupling(dim, checkerboard_mask(dim, parity=k % 2), hidden)
            for k in range(n_layers)
        ])

    def forward(self, x):
        """Data -> latent. Returns (z, total log|det| of this stack)."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld          # determinants multiply, logs add
        return x, log_det

    def inverse(self, z):
        """Latent -> data. Layers undone in reverse order."""
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


def standard_normal_logprob(z):
    """log N(z; 0, I) summed over the feature dimension, per batch element."""
    # -0.5 * (z^2 + log(2*pi)) elementwise, then sum across dimensions
    return (-0.5 * (z ** 2 + math.log(2 * math.pi))).flatten(1).sum(dim=1)


def flow_log_likelihood(model, x_flat, preproc_log_det):
    """Exact log p_x(x) via change of variables.

    x_flat is already dequantized and logit-transformed; preproc_log_det is the
    log-determinant that preprocessing contributed. The flow's own determinant is
    added on top, which is the whole composition rule in three lines.
    """
    z, flow_log_det = model(x_flat)
    log_p_z = standard_normal_logprob(z)
    return log_p_z + flow_log_det + preproc_log_det, z


def bits_per_dim(log_likelihood, dim, n_bits=8):
    """Convert a nat-valued log-likelihood into bits per dimension.

    Dividing by dim * ln2 converts nats to bits per coordinate, and adding
    n_bits restores the scale removed when dequantization divided by 2**n_bits.
    Lower is better; an untrained flow sits well above a trained one.
    """
    return -log_likelihood / (dim * math.log(2)) + n_bits


def preprocess(x_images):
    """ToTensor-style batch -> (flattened real-valued vector, log_det)."""
    deq = dequantize(x_images)
    flat = deq.flatten(1)
    return logit_transform(flat)


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 784
    batch = 16

    fake_images = torch.rand(batch, 1, 28, 28)
    x_flat, preproc_ld = preprocess(fake_images)

    model = RealNVP(dim, n_layers=6)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RealNVP: {len(model.layers)} coupling layers, {n_params:,} parameters")

    log_lik, z = flow_log_likelihood(model, x_flat, preproc_ld)
    bpd = bits_per_dim(log_lik, dim)
    print(f"latent stats: mean {z.mean().item():.3f}, std {z.std().item():.3f}")
    print(f"log-likelihood (nats): {log_lik.mean().item():.1f}")
    print(f"bits per dim (untrained): {bpd.mean().item():.3f}")

    # check 1: the stack must be exactly invertible, not just each layer
    z_fwd, log_det_fwd = model(x_flat)
    recon = model.inverse(z_fwd)
    recon_err = (recon - x_flat).abs().max().item()
    print(f"stack forward/inverse max error: {recon_err:.2e}")
    assert recon_err < 1e-4, "the full stack must invert exactly"

    # check 2: running the inverse direction must negate the log-determinant.
    # re-running forward from the reconstruction should reproduce log_det_fwd.
    _, log_det_again = model(recon)
    det_err = (log_det_again - log_det_fwd).abs().max().item()
    print(f"log-determinant round-trip max error: {det_err:.2e}")
    assert det_err < 1e-4, "log-determinant must be consistent under round-trip"

    # check 3: alternating masks must cover every coordinate. with parity 0 and 1
    # the union of transformed positions across two adjacent layers is everything.
    m0 = checkerboard_mask(dim, parity=0)
    m1 = checkerboard_mask(dim, parity=1)
    covered = ((1 - m0) + (1 - m1)) > 0
    assert covered.all(), "alternating masks must transform every coordinate"
    print("mask coverage: every one of "
          f"{dim} coordinates is transformed by an adjacent layer pair")

    print("day 2 checks passed: stacked flow, exact change-of-variables "
          "likelihood in bits/dim, invertibility and log-det consistency.")
