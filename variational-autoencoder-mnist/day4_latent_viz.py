"""
Day 4 (final): now that we can train a VAE (day 2) and tune the KL weight
(day 3), look at what the latent space actually learned. Three views:

  1. Latent traversal - take one encoded digit, walk a single latent dim
     across a range while holding the others fixed, and decode each step.
     A well-behaved dim should morph the digit smoothly (slant, thickness, ...).
  2. Prior sampling - draw z ~ N(0, I) and decode. Because training pulled
     q(z|x) toward the prior, samples from the prior should look like plausible
     digits, not noise.
  3. t-SNE of encoded means - embed the test set's mu vectors to 2D and color
     by label. Digits should cluster even though the VAE never saw the labels.

The model and loss are imported from day 2 so this file is purely the analysis
layer. We train briefly here for a self-contained run; in practice you'd load a
checkpoint from the day 2/3 training.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from day2_vae_elbo import (
    BATCH_SIZE,
    DEVICE,
    LATENT_DIM,
    SEED,
    VAE,
    elbo_loss,
)

DATA_ROOT = Path("./data")
PLOT_DIR = Path("./plots")
TRAVERSAL_RANGE = (-3.0, 3.0)  # in units of the standard-normal prior
TRAVERSAL_STEPS = 9
TRAIN_EPOCHS = 3  # short - just enough to get a usable latent for the viz


def get_loaders() -> Tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


def quick_train(model: VAE, loader: DataLoader, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            bce, kl = elbo_loss(recon, x, mu, logvar)
            loss = bce + kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"  pretrain epoch {ep}/{epochs}  elbo {running / len(loader):7.2f}")


def most_active_dims(model: VAE, loader: DataLoader, k: int = 3) -> List[int]:
    """Rank latent dims by the variance of their encoded means across the test
    set. Dims the posterior actually uses have high variance; collapsed dims
    sit near 0, so these are the interesting ones to traverse."""
    model.eval()
    mus = []
    with torch.no_grad():
        for x, _ in loader:
            mu, _ = model.encode(x.to(DEVICE))
            mus.append(mu.cpu())
    all_mu = torch.cat(mus, dim=0)
    var = all_mu.var(dim=0)
    return var.argsort(descending=True)[:k].tolist()


def latent_traversal(model: VAE, base_z: torch.Tensor, dim: int) -> torch.Tensor:
    """Decode `base_z` while sweeping a single latent dim across the range.
    Returns a (steps, 28, 28) stack of reconstructions."""
    model.eval()
    lo, hi = TRAVERSAL_RANGE
    values = torch.linspace(lo, hi, TRAVERSAL_STEPS)
    frames = []
    with torch.no_grad():
        for v in values:
            z = base_z.clone()
            z[0, dim] = v
            img = model.decode(z).view(28, 28).cpu()
            frames.append(img)
    return torch.stack(frames)


def sample_prior(model: VAE, n: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        return model.decode(z).view(n, 28, 28).cpu()


def encode_test_set(model: VAE, loader: DataLoader, limit: int = 2000):
    """Collect (mu, label) pairs for at most `limit` test points for t-SNE."""
    model.eval()
    mus, labels = [], []
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x.to(DEVICE))
            mus.append(mu.cpu())
            labels.append(y)
            seen += x.shape[0]
            if seen >= limit:
                break
    return torch.cat(mus)[:limit], torch.cat(labels)[:limit]


def save_visualizations(traversals, samples, mu, labels) -> None:
    """Render the three views into one figure. Imports are local so the rest of
    the module stays importable without a display/sklearn present."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib/sklearn not available - skipping plot render")
        return

    PLOT_DIR.mkdir(exist_ok=True)

    # traversals: one row per swept dim
    n_rows = len(traversals)
    fig, axes = plt.subplots(n_rows, TRAVERSAL_STEPS, figsize=(TRAVERSAL_STEPS, n_rows))
    for r, (dim, frames) in enumerate(traversals):
        for c in range(TRAVERSAL_STEPS):
            ax = axes[r][c] if n_rows > 1 else axes[c]
            ax.imshow(frames[c], cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(f"z[{dim}]", rotation=0, labelpad=20)
    fig.suptitle("latent traversals (most active dims)")
    fig.savefig(PLOT_DIR / "latent_traversals.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # prior samples in a square grid
    side = int(len(samples) ** 0.5)
    fig, axes = plt.subplots(side, side, figsize=(side, side))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    fig.suptitle("samples from the prior N(0, I)")
    fig.savefig(PLOT_DIR / "prior_samples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # t-SNE of encoded means
    emb = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(mu.numpy())
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels.numpy(), cmap="tab10", s=6, alpha=0.7)
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title("t-SNE of encoded means, colored by digit")
    fig.savefig(PLOT_DIR / "tsne_latent.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved 3 figures to {PLOT_DIR}/")


def main() -> None:
    torch.manual_seed(SEED)
    train_loader, test_loader = get_loaders()
    model = VAE().to(DEVICE)

    print("pretraining briefly for a usable latent space:")
    quick_train(model, train_loader, TRAIN_EPOCHS)

    dims = most_active_dims(model, test_loader, k=3)
    print(f"most active latent dims (by mu variance): {dims}")

    # use the first test image as the anchor for the traversals
    sample_x, _ = next(iter(test_loader))
    with torch.no_grad():
        base_mu, _ = model.encode(sample_x[:1].to(DEVICE))
    traversals = [(d, latent_traversal(model, base_mu, d)) for d in dims]

    samples = sample_prior(model, 64)
    mu, labels = encode_test_set(model, test_loader, limit=2000)

    save_visualizations(traversals, samples, mu, labels)
    print("project complete - VAE trained, latent space probed and visualized")


if __name__ == "__main__":
    main()
