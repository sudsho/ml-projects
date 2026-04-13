"""
Image Classification with CNNs - Day 2
Build a CNN from scratch with PyTorch for CIFAR-10 classification.
Architecture: Conv blocks with BatchNorm + Dropout, followed by FC layers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Config ---
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 channel stats (computed in Day 1)
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2470, 0.2435, 0.2616]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --- Data loaders ---
def get_dataloaders(batch_size=BATCH_SIZE):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transforms)
    test_set  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# --- Model ---
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional MaxPool."""
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))  # halves spatial dims
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CIFARNet(nn.Module):
    """
    Custom CNN for CIFAR-10 (32x32 input).

    Architecture:
      Block 1: 3  → 64  (no pool)
      Block 2: 64 → 128 (pool → 16x16)
      Block 3: 128→ 256 (no pool)
      Block 4: 256→ 256 (pool → 8x8)
      Block 5: 256→ 512 (pool → 4x4)
      FC: 512*4*4 → 1024 → 512 → 10
    """
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64,  pool=False),
            ConvBlock(64,  128, pool=True),   # 16x16
            ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=True),   # 8x8
            ConvBlock(256, 512, pool=True),   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --- Training utilities ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Training loop ---
def train(epochs=EPOCHS, lr=LR, device=DEVICE):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    train_loader, test_loader = get_dataloaders()
    model = CIFARNet(num_classes=10, dropout=0.4).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_cifarnet.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model, history


def plot_history(history, save_path="plots/cnn_training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("CIFARNet - Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Saved training curves to {save_path}")
    plt.close()


# --- Main ---
if __name__ == "__main__":
    print("=== CIFAR-10 CNN from Scratch - Day 2 ===\n")

    model, history = train(epochs=EPOCHS, lr=LR, device=DEVICE)
    plot_history(history)

    # Quick sanity check on final model
    train_loader, test_loader = get_dataloaders(batch_size=256)
    criterion = nn.CrossEntropyLoss()
    _, final_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print("\nDay 2 complete. Checkpoint saved to ./checkpoints/best_cifarnet.pt")
    print("Next: Training loop analysis and loss curve deep-dive (Day 3)")
