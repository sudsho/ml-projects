"""
Day 2: the actual VAE. We swap the deterministic bottleneck from day 1 for a
probabilistic one - the encoder now emits a mean and a log-variance, we sample
z with the reparameterization trick, and the loss becomes the ELBO: a
reconstruction term (BCE) plus the KL divergence between the approximate
posterior q(z|x) and the standard-normal prior.

Keeping the BCE comparable to the day 1 baseline lets us see what the KL term
costs us in raw reconstruction quality.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path("./data")
BATCH_SIZE = 128
LATENT_DIM = 16
HIDDEN = 256
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 17


@dataclass
class EpochStats:
    epoch: int
    bce: float
    kl: float

    @property
    def elbo(self) -> float:
        # we minimize bce + kl, so the (negative) ELBO is just their sum here
        return self.bce + self.kl


def get_loaders() -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor()])  # [0, 1] for the Bernoulli decoder
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


class VAE(nn.Module):
    """MLP VAE. Encoder maps to (mu, logvar); decoder maps a sampled z back to
    pixel probabilities. Architecture mirrors the day 1 baseline so the only
    real difference is the stochastic latent and the KL term."""

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + sigma * eps, eps ~ N(0, I). Doing it this way keeps the
        # sampling differentiable w.r.t. the encoder params.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon.view(x.shape), mu, logvar


def elbo_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (reconstruction, kl) both averaged per-sample so the numbers stay
    on the same scale as the day 1 BCE baseline."""
    batch = x.shape[0]
    # summed over pixels, averaged over the batch
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / batch
    # closed-form KL between N(mu, sigma^2) and N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return bce, kl


def run_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer | None) -> tuple[float, float]:
    is_train = opt is not None
    model.train(is_train)
    bce_hist: List[float] = []
    kl_hist: List[float] = []
    for x, _ in loader:
        x = x.to(DEVICE)
        recon, mu, logvar = model(x)
        bce, kl = elbo_loss(recon, x, mu, logvar)
        loss = bce + kl
        if is_train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        bce_hist.append(bce.item())
        kl_hist.append(kl.item())
    return statistics.mean(bce_hist), statistics.mean(kl_hist)


def main() -> None:
    torch.manual_seed(SEED)
    train_loader, val_loader = get_loaders()
    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history: List[EpochStats] = []
    for ep in range(1, EPOCHS + 1):
        tr_bce, tr_kl = run_epoch(model, train_loader, opt)
        with torch.no_grad():
            va_bce, va_kl = run_epoch(model, val_loader, None)
        history.append(EpochStats(ep, va_bce, va_kl))
        print(
            f"epoch {ep:2d} | train bce {tr_bce:7.2f} kl {tr_kl:6.2f} "
            f"| val bce {va_bce:7.2f} kl {va_kl:6.2f}"
        )

    last = history[-1]
    print(f"\nfinal val ELBO (bce+kl): {last.elbo:.2f}  [bce {last.bce:.2f} | kl {last.kl:.2f}]")
    print("next up (day 3): beta-VAE sweep and a KL annealing schedule to trade these off")


if __name__ == "__main__":
    main()
