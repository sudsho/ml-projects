"""
Day 4 - Transfer Learning with ResNet on CIFAR-10
Fine-tune a pre-trained ResNet-18 model from torchvision.
We replace the final FC layer and train only the classifier head first,
then unfreeze the entire network for full fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_EPOCHS_HEAD = 5      # train only classifier head
NUM_EPOCHS_FINETUNE = 10 # then fine-tune whole network
LR_HEAD = 1e-3
LR_FINETUNE = 5e-5
DATA_DIR = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Data transforms ───────────────────────────────────────────────────────────
# ResNet expects ImageNet stats; CIFAR-10 images are 32×32 → resize to 224
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=0),   # keep size after resize
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dataloaders():
    train_set = datasets.CIFAR10(root=DATA_DIR, train=True,
                                  download=True, transform=train_transform)
    val_set   = datasets.CIFAR10(root=DATA_DIR, train=False,
                                  download=True, transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train samples: {len(train_set):,}  |  Val samples: {len(val_set):,}")
    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────

def build_resnet18(freeze_backbone: bool = True) -> nn.Module:
    """Load ImageNet-pretrained ResNet-18 and adapt the final layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classification head
    in_features = model.fc.in_features          # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def run_training(model, train_loader, val_loader, num_epochs, lr, tag):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, f"resnet18_{tag}_best.pth"))

        print(f"[{tag}] Epoch {epoch:02d}/{num_epochs}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.2f}%")

    return history, best_val_acc


def plot_history(history_head, history_full, save_path):
    epochs_head = range(1, len(history_head["val_acc"]) + 1)
    epochs_full = range(len(epochs_head) + 1,
                        len(epochs_head) + len(history_full["val_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(list(epochs_head), history_head["train_loss"], "b-", label="Train (head)")
    axes[0].plot(list(epochs_head), history_head["val_loss"],   "b--", label="Val (head)")
    axes[0].plot(list(epochs_full), history_full["train_loss"], "r-", label="Train (full)")
    axes[0].plot(list(epochs_full), history_full["val_loss"],   "r--", label="Val (full)")
    axes[0].set_title("Loss Curve - ResNet-18 Transfer Learning")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(list(epochs_head), history_head["train_acc"], "b-", label="Train (head)")
    axes[1].plot(list(epochs_head), history_head["val_acc"],   "b--", label="Val (head)")
    axes[1].plot(list(epochs_full), history_full["train_acc"], "r-", label="Train (full)")
    axes[1].plot(list(epochs_full), history_full["val_acc"],   "r--", label="Val (full)")
    axes[1].set_title("Accuracy Curve - ResNet-18 Transfer Learning")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()

    # Phase 1 — train classifier head only (backbone frozen)
    print("\n=== Phase 1: Training classifier head (backbone frozen) ===")
    model = build_resnet18(freeze_backbone=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    history_head, best_head = run_training(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS_HEAD, lr=LR_HEAD, tag="head"
    )
    print(f"\nBest val accuracy (head-only): {best_head:.2f}%")

    # Phase 2 — unfreeze everything and fine-tune with a lower LR
    print("\n=== Phase 2: Full fine-tuning (all layers unfrozen) ===")
    for param in model.parameters():
        param.requires_grad = True

    # Load best head checkpoint before fine-tuning
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "resnet18_head_best.pth"),
                   map_location=DEVICE)
    )

    history_full, best_full = run_training(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS_FINETUNE, lr=LR_FINETUNE, tag="full"
    )
    print(f"\nBest val accuracy (full fine-tune): {best_full:.2f}%")

    # Plot combined training history
    plot_history(
        history_head, history_full,
        save_path=os.path.join(RESULTS_DIR, "resnet18_training_curves.png")
    )

    # Save results summary
    summary = {
        "model": "ResNet-18 (ImageNet pretrained)",
        "dataset": "CIFAR-10",
        "phase1_epochs": NUM_EPOCHS_HEAD,
        "phase2_epochs": NUM_EPOCHS_FINETUNE,
        "best_head_only_acc": round(best_head, 2),
        "best_finetune_acc": round(best_full, 2),
        "lr_head": LR_HEAD,
        "lr_finetune": LR_FINETUNE,
    }
    with open(os.path.join(RESULTS_DIR, "resnet18_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))
