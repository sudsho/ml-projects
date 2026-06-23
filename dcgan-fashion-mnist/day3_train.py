"""Day 3 - the full adversarial training loop, fixed-noise sampling, and monitoring.

Days 1 and 2 gave us the two networks and a single alternating update step. The
only thing left before sampling (day 4) is to run that step over the Fashion-MNIST
dataloader for several epochs and, crucially, to *watch* it: GAN training has no
single loss that monotonically decreases, so "is it working?" is answered by a
handful of diagnostic signals rather than a validation curve.

Two practices make a DCGAN run legible:

  - a *fixed* noise batch, drawn once before training and never resampled, that we
    push through the generator at the end of every epoch. Because the input is held
    constant, the resulting image grids isolate how G itself is improving - frame to
    frame you see the same 64 latent points sharpen from noise into garments.
  - per-step metric history (loss_D, loss_G, D_x, D(G(z)) before and after the G
    update) so day 4 can plot the curves and so this loop can flag the two classic
    failure modes early.

The optimiser recipe is the DCGAN one from day 2's sanity check: Adam, lr 2e-4,
with beta1 lowered from the usual 0.9 to 0.5. The high default momentum makes the
adversarial game oscillate; 0.5 noticeably stabilises it.
"""

import os

import torch

from day1_models import (
    Generator,
    Discriminator,
    weights_init,
    get_dataloader,
    LATENT_DIM,
)
from day2_losses import make_criterion, train_one_step


def make_optimizers(netD, netG, lr=2e-4, beta1=0.5):
    """Two independent Adam optimisers - one per network, never shared.

    Each net must descend its own loss, so they cannot share an optimiser or a
    gradient buffer. beta1 is the only departure from Adam's defaults: dropping the
    first-moment decay to 0.5 shortens the optimiser's memory of past gradients,
    which damps the back-and-forth swings that a fast-moving adversary induces.
    """
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    return optimizerD, optimizerG


def diagnose(metrics):
    """Translate one step's metrics into a mode-collapse / divergence verdict.

    Two failure modes dominate DCGAN training and both are visible in the D output
    averages:

      - discriminator wins: D_x -> 1 and D(G(z)) -> 0. D has memorised the reals,
        G's loss saturates, and the gradient handed back to G all but vanishes.
      - mode collapse: G(z) for many different z map to the same handful of images.
        D then drives D(G(z)) toward 0 again, but the tell is that D_G_z2 (after G's
        own update) stops recovering - G can no longer fool D because it has thrown
        away its diversity.

    Healthy training keeps D_x and D(G(z)) loosely around 0.5 and produces a visible
    gap between D_G_z1 (before G updates) and D_G_z2 (after).
    """
    flags = []
    if metrics["D_x"] > 0.99 and metrics["D_G_z1"] < 0.01:
        flags.append("discriminator overpowering G (D_x~1, D(G(z))~0)")
    if metrics["D_G_z2"] < 0.05 and metrics["loss_G"] > 5.0:
        flags.append("generator gradient starving (loss_G blowing up)")
    return flags


@torch.no_grad()
def sample_grid(netG, fixed_noise):
    """Generate the fixed-noise batch and arrange it as a single [0,1] image grid.

    Kept eval-mode and grad-free so BatchNorm uses its running stats and no graph is
    built. The generator emits [-1, 1] (its Tanh range), so we shift back to [0, 1]
    before handing the grid to torchvision for saving/plotting in day 4.
    """
    from torchvision.utils import make_grid

    netG.eval()
    fakes = netG(fixed_noise)
    netG.train()
    grid = make_grid(fakes, nrow=8, normalize=True, value_range=(-1, 1))
    return grid


def train(num_epochs=5, batch_size=128, sample_dir="samples", device=None,
          log_every=100, max_steps_per_epoch=None):
    """Run the full DCGAN training loop and return the metric history + sample grids.

    `max_steps_per_epoch` exists only so this module can be smoke-tested on CPU in
    seconds; leave it None for a real run over the whole dataset. Sample grids are
    captured in memory each epoch (and written to `sample_dir` when torchvision can)
    so day 4 can stitch them into a training-progression figure.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(sample_dir, exist_ok=True)

    netG = Generator().to(device).apply(weights_init)
    netD = Discriminator().to(device).apply(weights_init)
    criterion = make_criterion()
    optimizerD, optimizerG = make_optimizers(netD, netG)

    loader = get_dataloader(batch_size=batch_size, train=True)

    # the one noise batch that never changes - our window onto G's progress
    fixed_noise = torch.randn(64, LATENT_DIM, device=device)

    history = {k: [] for k in ("loss_D", "loss_G", "D_x", "D_G_z1", "D_G_z2")}
    grids = []

    for epoch in range(num_epochs):
        for step, (real, _) in enumerate(loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            real = real.to(device)
            metrics = train_one_step(
                (netD, netG), criterion, (optimizerD, optimizerG), real, device
            )
            for k in history:
                history[k].append(metrics[k])

            if step % log_every == 0:
                print(
                    f"[epoch {epoch + 1}/{num_epochs}][step {step}] "
                    f"loss_D={metrics['loss_D']:.3f} loss_G={metrics['loss_G']:.3f} "
                    f"D(x)={metrics['D_x']:.2f} "
                    f"D(G(z))={metrics['D_G_z1']:.2f}->{metrics['D_G_z2']:.2f}"
                )
                for flag in diagnose(metrics):
                    print(f"    [warn] {flag}")

        # end-of-epoch fixed-noise snapshot
        grid = sample_grid(netG, fixed_noise)
        grids.append(grid.cpu())
        try:
            from torchvision.utils import save_image
            save_image(grid, os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"))
        except Exception as err:  # saving is a convenience, not required to train
            print(f"    (could not write sample image: {err})")

    return {"netG": netG, "netD": netD, "history": history,
            "grids": grids, "fixed_noise": fixed_noise}


def _sanity_check():
    """A two-epoch, few-step run on whatever device is handy to prove the loop runs.

    This still downloads Fashion-MNIST on first call; the tiny step cap keeps it to a
    couple of optimiser steps so we are only checking that the loop wires together
    and that the history/grid bookkeeping is populated.
    """
    out = train(num_epochs=2, batch_size=32, max_steps_per_epoch=3, log_every=1)
    hist = out["history"]
    assert len(hist["loss_G"]) == len(hist["loss_D"]) > 0
    assert len(out["grids"]) == 2, "expected one sample grid per epoch"
    assert out["grids"][0].shape[0] in (1, 3), "grid should be an image tensor"
    assert all(torch.isfinite(torch.tensor(hist["loss_D"]))), "loss_D went non-finite"
    print(f"\nran {len(hist['loss_G'])} steps, captured {len(out['grids'])} grids")
    print("day 3 training loop OK")


if __name__ == "__main__":
    _sanity_check()
