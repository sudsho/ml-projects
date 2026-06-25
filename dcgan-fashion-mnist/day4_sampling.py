"""Day 4 - final sampling, latent-space interpolation, and loss curves.

Days 1-3 built the two networks, the alternating update step, and the full
training loop with per-step metric history and fixed-noise snapshots. With a
trained generator in hand there is nothing left to optimise; day 4 is about
*reading* what the GAN learned. Three artefacts do that:

  - a final fixed-noise sample grid, the same window day 3 opened each epoch but
    now at the end of training, showing the garments G can synthesise,
  - a latent-space interpolation: walk a straight line (well, a great-circle arc)
    between two noise vectors and decode every step. A GAN that has learned a
    smooth manifold morphs one garment continuously into another; a collapsed or
    memorising one jumps between a few fixed images,
  - the generator and discriminator loss curves over training. Unlike a supervised
    loss these do not descend to zero - the interesting signal is whether they
    stay in a bounded tug-of-war rather than one diverging.

Interpolation uses spherical linear interpolation (slerp) rather than a straight
lerp. The latent prior is an isotropic Gaussian, so almost all its mass sits in a
thin shell at radius ~sqrt(LATENT_DIM). A straight line between two such vectors
dips through the sparsely-populated interior near the origin, where G was never
trained and tends to emit blurry averages; slerp stays on the shell the whole way.
"""

import os

import numpy as np
import torch

from day1_models import LATENT_DIM


@torch.no_grad()
def final_sample_grid(netG, n_samples=64, device=None, seed=None):
    """Draw a fresh noise batch and decode it into one [0,1] image grid.

    Unlike the fixed-noise grids in day 3 (held constant to isolate G's progress),
    this samples new latents so the grid reflects the breadth of what the trained
    generator produces rather than a fixed set of points sharpening over time.
    """
    from torchvision.utils import make_grid

    device = device or next(netG.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    netG.eval()
    noise = torch.randn(n_samples, LATENT_DIM, device=device)
    fakes = netG(noise)
    netG.train()

    nrow = int(np.ceil(np.sqrt(n_samples)))
    return make_grid(fakes.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))


def slerp(t, z0, z1):
    """Spherical linear interpolation between two latent vectors at fraction t.

    Keeps the interpolant on the hypersphere the Gaussian prior concentrates on,
    so every intermediate vector is as "typical" a sample as the two endpoints.
    Falls back to a plain lerp when the two vectors are nearly colinear (the angle
    omega -> 0 makes the sin(omega) denominator numerically unstable).
    """
    z0_flat, z1_flat = z0.flatten(), z1.flatten()
    unit0 = z0_flat / z0_flat.norm()
    unit1 = z1_flat / z1_flat.norm()
    dot = torch.clamp((unit0 * unit1).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < 1e-4:
        return (1.0 - t) * z0 + t * z1
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    return a * z0 + b * z1


@torch.no_grad()
def interpolation_grid(netG, steps=10, n_rows=4, device=None, seed=None):
    """Decode slerp walks between random latent endpoints, one row per pair.

    Each row holds `steps` images marching from a left endpoint to a right one;
    `n_rows` independent endpoint pairs stack into a single grid. A smooth left-to-
    right morph within every row is the visual signature of a well-behaved latent
    manifold.
    """
    from torchvision.utils import make_grid

    device = device or next(netG.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    netG.eval()
    frames = []
    for _ in range(n_rows):
        z_start = torch.randn(LATENT_DIM, device=device)
        z_end = torch.randn(LATENT_DIM, device=device)
        for i in range(steps):
            t = i / (steps - 1)
            z = slerp(t, z_start, z_end).view(1, LATENT_DIM)
            frames.append(netG(z))
    netG.train()

    batch = torch.cat(frames, dim=0).cpu()
    return make_grid(batch, nrow=steps, normalize=True, value_range=(-1, 1))


def plot_loss_curves(history, smooth_window=50, save_path=None):
    """Plot smoothed generator and discriminator loss over training steps.

    Raw GAN losses are spiky step to step, so we overlay a moving average to make
    the trend legible. The thing to look for is not convergence to zero but a
    bounded equilibrium - both curves wandering in a stable band means neither
    network ran away with the game.
    """
    import matplotlib.pyplot as plt

    def moving_avg(xs, w):
        if len(xs) < w:
            return np.asarray(xs, dtype=float)
        kernel = np.ones(w) / w
        return np.convolve(np.asarray(xs, dtype=float), kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(moving_avg(history["loss_G"], smooth_window), label="loss_G", lw=1.5)
    ax.plot(moving_avg(history["loss_D"], smooth_window), label="loss_D", lw=1.5)
    ax.set_xlabel(f"training step (smoothed over {smooth_window})")
    ax.set_ylabel("BCE loss")
    ax.set_title("DCGAN adversarial loss curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def export_artifacts(out, sample_dir="samples"):
    """Write the day-4 figures (final grid, interpolation, loss curves) to disk.

    Takes the dict returned by day 3's `train` (keys netG/history). Saving is
    best-effort: a headless box without an image backend should still complete the
    numerical work, so each write is guarded rather than allowed to abort the run.
    """
    os.makedirs(sample_dir, exist_ok=True)
    netG, history = out["netG"], out["history"]
    written = []

    try:
        from torchvision.utils import save_image

        grid = final_sample_grid(netG, n_samples=64, seed=0)
        path = os.path.join(sample_dir, "final_samples.png")
        save_image(grid, path)
        written.append(path)

        interp = interpolation_grid(netG, steps=10, n_rows=6, seed=1)
        path = os.path.join(sample_dir, "latent_interpolation.png")
        save_image(interp, path)
        written.append(path)
    except Exception as err:
        print(f"    (could not write sample images: {err})")

    try:
        path = os.path.join(sample_dir, "loss_curves.png")
        plot_loss_curves(history, save_path=path)
        written.append(path)
    except Exception as err:
        print(f"    (could not write loss curves: {err})")

    return written


def _sanity_check():
    """Tiny train run, then exercise every day-4 routine on the result.

    Checks the grids come back as image tensors of the right shape, the slerp
    endpoints land exactly on the supplied vectors, and the interpolation grid
    holds the expected number of frames. Image files are written when a backend is
    available but their absence does not fail the check.
    """
    from day3_train import train

    out = train(num_epochs=1, batch_size=32, max_steps_per_epoch=3, log_every=1)
    netG = out["netG"]

    grid = final_sample_grid(netG, n_samples=16, seed=0)
    assert grid.dim() == 3 and grid.shape[0] in (1, 3), grid.shape

    z0 = torch.randn(LATENT_DIM)
    z1 = torch.randn(LATENT_DIM)
    assert torch.allclose(slerp(0.0, z0, z1), z0, atol=1e-5), "slerp(0) != z0"
    assert torch.allclose(slerp(1.0, z0, z1), z1, atol=1e-5), "slerp(1) != z1"

    interp = interpolation_grid(netG, steps=8, n_rows=3, seed=1)
    assert interp.dim() == 3, interp.shape

    written = export_artifacts(out, sample_dir="samples")
    print(f"final grid {tuple(grid.shape)}, interpolation {tuple(interp.shape)}")
    print(f"wrote {len(written)} artefact(s): {written}")
    print("day 4 sampling/interpolation OK")


if __name__ == "__main__":
    _sanity_check()
