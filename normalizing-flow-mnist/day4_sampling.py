"""
Day 4 of the RealNVP normalizing flow on MNIST.

Days 1-3 built the flow and trained it. Today is the payoff day: what the model
can do once the density is fitted, and what makes a flow different from the other
generative models in this repo.

  1. Sampling with a temperature knob. Generation is one inverse pass over
     z ~ N(0, I) - no ancestral chain like the DDPM, no discriminator like the
     DCGAN. Scaling the latent by T < 1 concentrates the draw toward the mode of
     the base density and cleans up samples, at the cost of diversity. Worth
     being precise about what that means: the samples are no longer draws from
     the model distribution, so a temperature grid is a picture of the model but
     not evidence about its likelihood.

  2. Latent interpolation, in two forms. Between two random latents, and between
     two *real* digits pushed through the forward map. The second one is where
     invertibility earns its keep - the flow encodes an actual image to the exact
     latent that reproduces it, so the interpolation endpoints are the real
     digits rather than blurry reconstructions of them. A VAE cannot do that; its
     encoder returns a distribution and the round trip loses information.

  3. Spherical interpolation rather than linear. In D = 784 dimensions a standard
     Gaussian puts essentially all its mass on a thin shell of radius ~sqrt(D).
     The straight line between two typical latents dips toward the origin, whose
     norm is far below that shell, so midpoints land in a region the model never
     saw during training and decode to washed-out digits. slerp keeps the norm
     roughly constant along the path and stays on the shell. Both are here so the
     difference is visible instead of asserted.

  4. The bits-per-dimension curves from training, since bpd is a real
     held-out number for a flow and not a bound. Saved with --plot.

Everything but the plot runs on CPU in a couple of minutes at the default epoch
count. Outputs land in samples/ and are gitignored.
"""

import argparse

import torch
from torchvision.utils import save_image

from day2_flow_model import RealNVP, bits_per_dim, flow_log_likelihood, preprocess
from day3_train import (
    DIM,
    N_BITS,
    SAMPLE_DIR,
    evaluate_bpd,
    get_loaders,
    inverse_preprocess,
    train,
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_at_temperature(model, n=64, temperature=1.0, device="cpu"):
    """Draw n samples with the base density scaled by `temperature`.

    z ~ N(0, T^2 I) instead of N(0, I). T = 1 is the honest model sample; below 1
    the draw is pulled toward the mode and looks cleaner but under-represents the
    tails, above 1 it exaggerates them. The flow itself is untouched - this is a
    change of the base distribution at sampling time only.
    """
    model.eval()
    z = torch.randn(n, DIM, device=device) * temperature
    return inverse_preprocess(model.inverse(z))


@torch.no_grad()
def temperature_sweep(model, temperatures=(0.6, 0.8, 1.0), n_per=8, device="cpu"):
    """One row of samples per temperature, sharing the same noise draw.

    Reusing a single z across the row is what makes the comparison readable: each
    column is the same latent direction seen at different scales, so any change
    down a column is the temperature and not a different sample.
    """
    model.eval()
    base = torch.randn(n_per, DIM, device=device)
    rows = [inverse_preprocess(model.inverse(base * t)) for t in temperatures]
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------------------
# Latent interpolation
# ---------------------------------------------------------------------------
def lerp(z0, z1, steps=10):
    """Straight-line interpolation. Included as the thing slerp improves on."""
    ts = torch.linspace(0, 1, steps, device=z0.device).unsqueeze(1)
    return (1 - ts) * z0.unsqueeze(0) + ts * z1.unsqueeze(0)


def slerp(z0, z1, steps=10, eps=1e-7):
    """Spherical interpolation along the great circle between two latents.

    Interpolating the angle keeps every intermediate point at roughly the same
    norm as the endpoints, which is what keeps the path on the typical-set shell
    where the model has density. Falls back to lerp when the endpoints are nearly
    parallel and sin(omega) would divide by ~0.
    """
    ts = torch.linspace(0, 1, steps, device=z0.device).unsqueeze(1)
    u0 = z0 / z0.norm()
    u1 = z1 / z1.norm()
    omega = torch.acos((u0 * u1).sum().clamp(-1 + eps, 1 - eps))
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < eps:
        return lerp(z0, z1, steps)
    a = torch.sin((1 - ts) * omega) / sin_omega
    b = torch.sin(ts * omega) / sin_omega
    return a * z0.unsqueeze(0) + b * z1.unsqueeze(0)


@torch.no_grad()
def interpolate_random(model, steps=10, device="cpu"):
    """A slerp row and a lerp row between the same pair of random latents."""
    model.eval()
    z0, z1 = torch.randn(2, DIM, device=device)
    path = torch.cat([slerp(z0, z1, steps), lerp(z0, z1, steps)], dim=0)
    return inverse_preprocess(model.inverse(path))


@torch.no_grad()
def interpolate_real(model, loader, steps=10, device="cpu"):
    """Interpolate between two real digits by encoding them first.

    The forward map is exact, so `model.inverse(model(x)[0])` returns x rather
    than an approximation of it. That makes the two ends of this row genuine
    dataset images, and everything between them a walk through the density the
    model learned.
    """
    model.eval()
    images, labels = next(iter(loader))
    images = images[:2].to(device)
    x_flat, _ = preprocess(images)
    z, _ = model(x_flat)

    path = slerp(z[0], z[1], steps)
    row = inverse_preprocess(model.inverse(path))
    endpoints = images.view(-1, 1, 28, 28)
    return torch.cat([endpoints[:1], row, endpoints[1:2]], dim=0), labels[:2].tolist()


@torch.no_grad()
def reconstruction_error(model, loader, device="cpu"):
    """Max |x - inverse(forward(x))| over one batch. Should be ~1e-5, not ~1e-1."""
    model.eval()
    images, _ = next(iter(loader))
    x_flat, _ = preprocess(images.to(device))
    z, _ = model(x_flat)
    return (model.inverse(z) - x_flat).abs().max().item()


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
def plot_bpd(history, path="bpd_curves.png"):
    """Train/val bits-per-dim against epoch. Imported lazily so the rest runs
    without matplotlib installed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [h[1] for h in history], marker="o", label="train")
    ax.plot(epochs, [h[2] for h in history], marker="s", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("bits per dimension")
    ax.set_title("RealNVP on MNIST - exact held-out likelihood")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10, help="interpolation steps")
    parser.add_argument("--plot", action="store_true", help="save the bpd curves")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, history = train(epochs=args.epochs)
    _, test_loader = get_loaders()

    final_bpd = evaluate_bpd(model, test_loader, device)
    print(f"final held-out bpd: {final_bpd:.4f}")

    err = reconstruction_error(model, test_loader, device)
    print(f"encode/decode round-trip max error: {err:.2e}")
    assert err < 1e-3, "a trained flow must still be exactly invertible"

    grid = sample_at_temperature(model, n=64, temperature=1.0, device=device)
    save_image(grid, f"{SAMPLE_DIR}/final_samples.png", nrow=8)

    sweep = temperature_sweep(model, device=device)
    save_image(sweep, f"{SAMPLE_DIR}/temperature_sweep.png", nrow=8)

    walk = interpolate_random(model, steps=args.steps, device=device)
    save_image(walk, f"{SAMPLE_DIR}/interpolation_random.png", nrow=args.steps)

    real_walk, digits = interpolate_real(model, test_loader, args.steps, device)
    save_image(real_walk, f"{SAMPLE_DIR}/interpolation_real.png",
               nrow=args.steps + 2)
    print(f"interpolated between a {digits[0]} and a {digits[1]}")

    if args.plot:
        plot_bpd(history)

    print("day 4 done: temperature sweep, slerp vs lerp walks, real-image "
          f"interpolation, {len(history)} epochs of bpd curves.")
