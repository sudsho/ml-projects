"""
Day 3: now that the plain VAE trains (day 2), poke at the reconstruction-vs-KL
tradeoff directly. Two knobs:

  1. beta-VAE: scale the KL term by a constant beta. beta > 1 pushes the model
     toward a more disentangled / more standard-normal latent at the cost of
     reconstruction; beta < 1 does the opposite.
  2. KL annealing: ramp beta from ~0 up to its target over the first few epochs.
     Early on the decoder can learn to use z before the KL term clamps the
     posterior down to the prior, which helps avoid posterior collapse.

We sweep a few betas, train each model briefly, and dump a reconstruction-vs-KL
tradeoff plot so the curve is easy to eyeball. The VAE/loss machinery is the
same as day 2 - only the weighting of the KL term changes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path("./data")
PLOT_DIR = Path("./plots")
BATCH_SIZE = 128
LATENT_DIM = 16
HIDDEN = 256
EPOCHS = 5
ANNEAL_EPOCHS = 3  # epochs over which beta ramps from 0 -> target
LR = 1e-3
BETAS = [0.5, 1.0, 2.0, 4.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 17


@dataclass
class RunResult:
    beta: float
    annealed: bool
    bce_history: List[float] = field(default_factory=list)
    kl_history: List[float] = field(default_factory=list)

    @property
    def final_bce(self) -> float:
        return self.bce_history[-1]

    @property
    def final_kl(self) -> float:
        return self.kl_history[-1]


def get_loaders() -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor()])  # [0, 1] for the Bernoulli decoder
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


class VAE(nn.Module):
    """Same MLP VAE as day 2. Kept here standalone so this script runs on its
    own without importing the day 2 module."""

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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon.view(x.shape), mu, logvar


def split_terms(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruction (BCE) and KL, both averaged per-sample. The beta weighting
    is applied by the caller so the raw terms stay comparable across runs."""
    batch = x.shape[0]
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return bce, kl


def beta_for_epoch(target_beta: float, epoch: int, annealed: bool) -> float:
    """Effective beta for a given epoch (1-indexed). With annealing we ramp
    linearly from 0 to the target over ANNEAL_EPOCHS, then hold."""
    if not annealed:
        return target_beta
    ramp = min(1.0, epoch / ANNEAL_EPOCHS)
    return target_beta * ramp


def run_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer | None, beta: float
) -> tuple[float, float]:
    is_train = opt is not None
    model.train(is_train)
    bce_hist: List[float] = []
    kl_hist: List[float] = []
    for x, _ in loader:
        x = x.to(DEVICE)
        recon, mu, logvar = model(x)
        bce, kl = split_terms(recon, x, mu, logvar)
        loss = bce + beta * kl
        if is_train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        bce_hist.append(bce.item())
        kl_hist.append(kl.item())
    return statistics.mean(bce_hist), statistics.mean(kl_hist)


def train_one(
    train_loader: DataLoader, val_loader: DataLoader, beta: float, annealed: bool
) -> RunResult:
    torch.manual_seed(SEED)  # same init across runs so beta is the only variable
    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    result = RunResult(beta=beta, annealed=annealed)

    for ep in range(1, EPOCHS + 1):
        eff_beta = beta_for_epoch(beta, ep, annealed)
        run_epoch(model, train_loader, opt, eff_beta)
        with torch.no_grad():
            # evaluate the raw terms at the target beta, not the ramped one
            va_bce, va_kl = run_epoch(model, val_loader, None, beta)
        result.bce_history.append(va_bce)
        result.kl_history.append(va_kl)
        tag = "anneal" if annealed else "fixed "
        print(
            f"  beta {beta:.1f} [{tag}] epoch {ep:2d} "
            f"(eff_beta {eff_beta:.2f}) | val bce {va_bce:7.2f} kl {va_kl:6.2f}"
        )
    return result


def save_tradeoff_plot(results: List[RunResult]) -> None:
    """Scatter final reconstruction (BCE) against final KL for every run. Lower
    BCE is better reconstruction; the betas trace out the tradeoff frontier."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping tradeoff plot")
        return

    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for annealed, marker in [(False, "o"), (True, "^")]:
        subset = [r for r in results if r.annealed == annealed]
        if not subset:
            continue
        xs = [r.final_kl for r in subset]
        ys = [r.final_bce for r in subset]
        label = "KL annealed" if annealed else "fixed beta"
        ax.plot(xs, ys, marker=marker, linestyle="--", label=label)
        for r in subset:
            ax.annotate(f"β={r.beta:g}", (r.final_kl, r.final_bce),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_xlabel("KL divergence (nats / sample)")
    ax.set_ylabel("reconstruction BCE (lower = better)")
    ax.set_title("VAE reconstruction vs KL tradeoff")
    ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / "beta_tradeoff.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved tradeoff plot -> {out}")


def main() -> None:
    train_loader, val_loader = get_loaders()
    results: List[RunResult] = []

    print("== fixed-beta sweep ==")
    for beta in BETAS:
        results.append(train_one(train_loader, val_loader, beta, annealed=False))

    print("\n== KL-annealed runs (target beta held after ramp) ==")
    for beta in [1.0, 4.0]:  # only the two that benefit most from a warmup
        results.append(train_one(train_loader, val_loader, beta, annealed=True))

    print("\nsummary (final epoch):")
    for r in results:
        tag = "anneal" if r.annealed else "fixed"
        print(f"  beta {r.beta:.1f} [{tag:6s}] bce {r.final_bce:7.2f} | kl {r.final_kl:6.2f}")

    save_tradeoff_plot(results)
    print("\nnext up (day 4): latent traversals, prior sampling, t-SNE on encoded digits")


if __name__ == "__main__":
    main()
