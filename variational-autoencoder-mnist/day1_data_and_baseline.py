"""
Day 1: load MNIST, set up the dataloaders, and train a plain (non-variational)
autoencoder as a baseline. The point is just to make sure the data pipeline
works and we have a reference reconstruction quality to beat once the VAE
machinery (KL term, reparameterization) shows up on day 2.
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
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 17


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float


def get_loaders() -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor()])  # values land in [0, 1]
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


class VanillaAE(nn.Module):
    """Plain MLP autoencoder, no probabilistic anything. Single hidden layer
    on each side keeps it deliberately small so the VAE has room to improve."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),  # back to [0, 1] for BCE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(x.shape)


def run_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer | None) -> float:
    is_train = opt is not None
    model.train(is_train)
    losses: List[float] = []
    for x, _ in loader:
        x = x.to(DEVICE)
        recon = model(x)
        loss = F.binary_cross_entropy(recon, x, reduction="mean")
        if is_train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return statistics.mean(losses)


def main() -> None:
    torch.manual_seed(SEED)
    train_loader, val_loader = get_loaders()
    model = VanillaAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history: List[EpochStats] = []
    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, opt)
        with torch.no_grad():
            vl = run_epoch(model, val_loader, None)
        history.append(EpochStats(ep, tr, vl))
        print(f"epoch {ep:2d} | train bce {tr:.4f} | val bce {vl:.4f}")

    # baseline number to beat with the actual VAE on day 3
    final_val = history[-1].val_loss
    print(f"\nvanilla AE baseline val BCE: {final_val:.4f}")
    print("(VAE on day 3 should be roughly comparable plus a KL term we measure separately)")


if __name__ == "__main__":
    main()
