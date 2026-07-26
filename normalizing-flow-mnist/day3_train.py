"""
Day 3 of the RealNVP normalizing flow on MNIST.

Day 2 closed the loop on the likelihood, which means training needs no new loss
function: a flow is trained by maximum likelihood, and the exact log p_x(x) is
already available. So the objective is literally "minimize bits per dimension",
which is the negative log-likelihood in a rescaled unit. That is unusual and
worth appreciating - there is no ELBO gap like a VAE, no adversarial game like a
GAN, no variational bound to loosen. The number printed during training is the
real held-out compression rate of the model.

Three things make the difference between a flow that trains and one that
diverges in the first hundred steps, and all three are here:

  1. Fresh dequantization noise every epoch. The noise is part of the *data
     distribution*, not a fixed preprocessing artifact, so it has to be resampled
     each time an image is visited. Caching one noisy copy lets the model
     memorize the noise and the reported bpd drifts below the honest value.

  2. Gradient clipping. The coupling layers' log-determinant term is unbounded
     below - the model can lower the loss by driving log_scale very negative,
     which concentrates density and produces enormous gradients. Clipping the
     global norm is what keeps that from turning into a NaN. The scale_factor /
     tanh parameterization from day 1 is the other half of that defence.

  3. Watching sample quality, not just loss. bpd falls monotonically long after
     samples stop improving, so a periodic sample grid is the honest diagnostic.
     Sampling is free here: draw z ~ N(0, I), run the flow backwards, invert the
     preprocessing. No iterative denoising, no decoder - one pass.

Validation bpd is computed with the model in eval mode over a fixed number of
batches. There is no dropout or batchnorm in this flow so eval mode changes
nothing numerically, but it stays in as the habit that catches the version of
this file where a batchnorm coupling variant gets added.
"""

import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from day2_flow_model import (
    RealNVP,
    bits_per_dim,
    flow_log_likelihood,
    preprocess,
)

DATA_DIR = "data"
SAMPLE_DIR = "samples"
DIM = 784
N_BITS = 8


def get_loaders(batch_size=128):
    """MNIST train/test loaders yielding raw [0,1] tensors.

    Deliberately no normalization here - dequantization and the logit transform
    are the model's own preprocessing and happen per batch inside the loop, so
    the loader hands over untouched ToTensor output.
    """
    tf = transforms.ToTensor()
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def inverse_preprocess(y, alpha=1e-6, n_bits=8):
    """Undo logit + dequantization so a latent sample becomes a viewable image.

    Exactly the inverse of day 1's pipeline: sigmoid back into the squeezed
    interval, un-squeeze to (0,1), then clamp. The clamp is cosmetic - an
    untrained flow can push samples slightly outside the cube and save_image
    would wrap those pixels.
    """
    s = torch.sigmoid(y)
    x = (s - alpha) / (1 - 2 * alpha)
    return x.clamp(0.0, 1.0).view(-1, 1, 28, 28)


@torch.no_grad()
def sample_grid(model, n=64, device="cpu"):
    """Draw n samples from the base Gaussian and push them through the inverse."""
    model.eval()
    z = torch.randn(n, DIM, device=device)
    x = model.inverse(z)
    return inverse_preprocess(x)


@torch.no_grad()
def evaluate_bpd(model, loader, device="cpu", max_batches=20):
    """Mean bits/dim over up to max_batches of held-out data."""
    model.eval()
    total, count = 0.0, 0
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        # fresh noise on the validation pass too - the honest number is an
        # expectation over the dequantization noise, not one lucky draw
        x_flat, preproc_ld = preprocess(images)
        log_lik, _ = flow_log_likelihood(model, x_flat, preproc_ld)
        bpd = bits_per_dim(log_lik, DIM, N_BITS)
        total += bpd.sum().item()
        count += bpd.shape[0]
    return total / max(count, 1)


def train(epochs=10, batch_size=128, lr=1e-3, clip_norm=5.0, sample_every=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    train_loader, test_loader = get_loaders(batch_size)
    model = RealNVP(DIM, n_layers=6, hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # flows are sensitive to the late-training learning rate; a plain cosine
    # decay to zero was enough here without any warmup
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"device {device}, {sum(p.numel() for p in model.parameters()):,} params")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen, clipped = 0.0, 0, 0

        for images, _ in train_loader:
            images = images.to(device)
            x_flat, preproc_ld = preprocess(images)   # new noise every visit
            log_lik, _ = flow_log_likelihood(model, x_flat, preproc_ld)

            # maximum likelihood == minimizing bits per dim, up to a constant
            loss = bits_per_dim(log_lik, DIM, N_BITS).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip_grad_norm_ returns the PRE-clip norm, so comparing it to the
            # threshold tells us how often the log-det term is blowing up
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if grad_norm > clip_norm:
                clipped += 1
            opt.step()

            running += loss.item() * images.shape[0]
            seen += images.shape[0]

        sched.step()
        train_bpd = running / seen
        val_bpd = evaluate_bpd(model, test_loader, device)
        history.append((epoch, train_bpd, val_bpd))
        print(f"epoch {epoch:2d} | train {train_bpd:.4f} bpd | "
              f"val {val_bpd:.4f} bpd | clipped {clipped} batches | "
              f"lr {sched.get_last_lr()[0]:.2e}")

        if epoch % sample_every == 0:
            grid = sample_grid(model, n=64, device=device)
            save_image(grid, f"{SAMPLE_DIR}/epoch_{epoch:02d}.png", nrow=8)

    return model, history


if __name__ == "__main__":
    torch.manual_seed(0)

    # a bpd sanity floor before touching real data: an untrained flow is roughly
    # the logit-space Gaussian fit, and anything below ~1.0 or above ~30 means
    # the change-of-variables bookkeeping is wrong rather than the model bad
    model = RealNVP(DIM, n_layers=6)
    fake = torch.rand(32, 1, 28, 28)
    x_flat, ld = preprocess(fake)
    log_lik, _ = flow_log_likelihood(model, x_flat, ld)
    start_bpd = bits_per_dim(log_lik, DIM, N_BITS).mean().item()
    print(f"untrained bpd on uniform noise: {start_bpd:.3f}")
    assert math.isfinite(start_bpd), "likelihood must be finite before training"

    # sampling has to work at init too, otherwise epoch-1 grids fail after a
    # long training run rather than in the first second
    grid = sample_grid(model, n=8)
    assert grid.shape == (8, 1, 28, 28), "samples must come back as images"
    assert (grid >= 0).all() and (grid <= 1).all(), "samples must be viewable"
    print(f"sample grid shape {tuple(grid.shape)}, "
          f"range [{grid.min():.3f}, {grid.max():.3f}]")

    trained, history = train(epochs=10)
    first, last = history[0][2], history[-1][2]
    print(f"val bpd {first:.4f} -> {last:.4f} over {len(history)} epochs")
