"""Day 2 - adversarial losses, label conventions, and the alternating train step.

Day 1 gave us a generator G and discriminator D with matching shapes. The pieces
missing before a full training loop (day 3) are the *objective* and the *update
rule*. A GAN is a two-player min-max game

    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

but in practice we never optimise that joint objective directly. Instead each
network gets its own loss and its own optimiser step, and they alternate. This
module implements one such alternating step plus the small but important details
that keep early training from collapsing: a single BCEWithLogits criterion, the
real=1 / fake=0 label convention, one-sided label smoothing, and the
non-saturating generator loss.

Everything here works on the *logits* D returns (D has no final sigmoid), so we
use BCEWithLogits, which folds the sigmoid into the loss in a numerically stable
log-sum-exp form. Feeding raw probabilities through a plain BCE would risk
log(0) once D gets confident.
"""

import torch
import torch.nn as nn

from day1_models import Generator, Discriminator, weights_init, LATENT_DIM

# Label values. Reals are labelled 1, fakes 0. One-sided label smoothing replaces
# the real target of 1.0 with 0.9: it discourages the discriminator from becoming
# over-confident on reals (which would shrink the gradient it hands back to G) and
# is the "one-sided" variant from Salimans et al. 2016 - we deliberately do NOT
# smooth the fake label, since smoothing both sides can reward G for matching the
# wrong density.
REAL_LABEL = 0.9
FAKE_LABEL = 0.0


def make_criterion():
    """Binary cross-entropy on logits - the shared loss for both networks."""
    return nn.BCEWithLogitsLoss()


def discriminator_step(netD, netG, criterion, optimizerD, real_batch, device):
    """One discriminator update: push D(real)->1 and D(fake)->0.

    The discriminator sees two minibatches per step - the real images and a fresh
    batch of fakes from G. Crucially we detach the fakes before they enter D:
    this step must not send gradients back into the generator, and detaching cuts
    the graph at G's output so only D's parameters move. The two half-losses are
    summed; their gradients accumulate into D's .grad before a single optimiser
    step.
    """
    batch_size = real_batch.size(0)
    optimizerD.zero_grad(set_to_none=True)

    # --- real half: target label 0.9 (smoothed) ---
    real_labels = torch.full((batch_size,), REAL_LABEL, device=device)
    logits_real = netD(real_batch)
    loss_real = criterion(logits_real, real_labels)
    loss_real.backward()

    # --- fake half: target label 0, detached so G is untouched ---
    noise = torch.randn(batch_size, LATENT_DIM, device=device)
    fake = netG(noise)
    fake_labels = torch.full((batch_size,), FAKE_LABEL, device=device)
    logits_fake = netD(fake.detach())
    loss_fake = criterion(logits_fake, fake_labels)
    loss_fake.backward()

    optimizerD.step()

    # D(x) averages for monitoring: ~0.5/0.5 is healthy, 1.0/0.0 means D is winning
    return {
        "loss_D": (loss_real + loss_fake).item(),
        "D_x": torch.sigmoid(logits_real).mean().item(),
        "D_G_z1": torch.sigmoid(logits_fake).mean().item(),
    }


def generator_step(netD, netG, criterion, optimizerG, batch_size, device):
    """One generator update using the non-saturating loss.

    The textbook min-max objective asks G to *minimise* log(1 - D(G(z))). Early on
    D rejects G's samples easily, that term saturates, and G's gradient vanishes.
    The standard fix (Goodfellow 2014) is to instead *maximise* log D(G(z)), which
    we get for free by relabelling the fakes as REAL (target 1) and reusing the
    same BCE criterion. Same optimum, far healthier gradients while G is weak.

    Note there is no detach here: we regenerate the fakes (or could reuse them)
    and let gradients flow all the way through G. D's parameters are frozen for
    this step only by virtue of us not calling optimizerD.step().
    """
    optimizerG.zero_grad(set_to_none=True)

    noise = torch.randn(batch_size, LATENT_DIM, device=device)
    fake = netG(noise)
    # G wants D to think these are real, so the target is the (smoothed) real label
    target = torch.full((batch_size,), REAL_LABEL, device=device)
    logits = netD(fake)
    loss_G = criterion(logits, target)
    loss_G.backward()
    optimizerG.step()

    return {
        "loss_G": loss_G.item(),
        "D_G_z2": torch.sigmoid(logits).mean().item(),
    }


def train_one_step(nets, criterion, optimizers, real_batch, device):
    """Full alternating step: update D once, then G once, on the same real batch.

    Returns a merged metrics dict. D_G_z1 (before G's update) should be low and
    D_G_z2 (after) higher - that gap is G learning to fool the current D within
    the step. The day 3 loop just calls this over the dataloader for N epochs.
    """
    netD, netG = nets
    optimizerD, optimizerG = optimizers

    d_metrics = discriminator_step(
        netD, netG, criterion, optimizerD, real_batch, device
    )
    g_metrics = generator_step(
        netD, netG, criterion, optimizerG, real_batch.size(0), device
    )
    return {**d_metrics, **g_metrics}


def _sanity_check():
    """One synthetic step to confirm losses are finite and both nets move."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    netG = Generator().to(device).apply(weights_init)
    netD = Discriminator().to(device).apply(weights_init)
    criterion = make_criterion()

    # DCGAN optimiser recipe: Adam, lr 2e-4, beta1 lowered to 0.5
    optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # snapshot one G weight so we can confirm the generator actually updated
    before = next(netG.parameters()).clone()

    fake_reals = torch.randn(16, 1, 32, 32, device=device)  # stand-in for a batch
    metrics = train_one_step(
        (netD, netG), criterion, (optimizerD, optimizerG), fake_reals, device
    )

    after = next(netG.parameters())
    moved = not torch.equal(before, after)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert moved, "generator parameters did not change after a step"

    print("step metrics:")
    for k, v in metrics.items():
        print(f"  {k:8s} = {v:.4f}")
    print(f"generator updated: {moved}")


if __name__ == "__main__":
    _sanity_check()
