"""
Image Classification with CNNs - Day 1
Data loading, exploration, and basic augmentation setup for CIFAR-10 dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# --- Config ---
DATA_DIR = "./data"
BATCH_SIZE = 64
NUM_WORKERS = 0  # set to 2+ on Linux/Mac
RANDOM_SEED = 42

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --- 1. Load raw data (no augmentation yet) ---
def load_raw_dataset():
    """Download and load CIFAR-10 with only tensor conversion."""
    base_transform = transforms.Compose([
        transforms.ToTensor(),  # scales to [0, 1]
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=base_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=base_transform
    )
    return train_dataset, test_dataset


# --- 2. Dataset statistics ---
def compute_channel_stats(dataset, num_samples=5000):
    """Compute per-channel mean and std over a subset of images."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0

    for i, (images, _) in enumerate(loader):
        if i * 256 >= num_samples:
            break
        # images shape: (B, C, H, W)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / total_pixels
    std = ((channel_sq_sum / total_pixels) - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def class_distribution(dataset):
    """Count samples per class."""
    labels = [label for _, label in dataset]
    counts = Counter(labels)
    return {CIFAR10_CLASSES[k]: v for k, v in sorted(counts.items())}


# --- 3. Visualization ---
def visualize_samples(dataset, n_per_class=5, save_path="plots/sample_grid.png"):
    """Plot a grid of n_per_class samples for each CIFAR-10 class."""
    import os
    os.makedirs("plots", exist_ok=True)

    class_images = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(class_images[label]) < n_per_class:
            class_images[label].append(img)
        if all(len(v) == n_per_class for v in class_images.values()):
            break

    fig = plt.figure(figsize=(n_per_class * 1.8, 10 * 1.8))
    gs = gridspec.GridSpec(10, n_per_class, figure=fig)

    for row in range(10):
        for col in range(n_per_class):
            ax = fig.add_subplot(gs[row, col])
            img = class_images[row][col].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CIFAR10_CLASSES[row], fontsize=9, rotation=0,
                              labelpad=40, va="center")

    fig.suptitle("CIFAR-10 Sample Images (5 per class)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    print(f"Saved sample grid to {save_path}")
    plt.close()


def plot_class_distribution(dist, save_path="plots/class_distribution.png"):
    """Bar plot of class counts."""
    import os
    os.makedirs("plots", exist_ok=True)

    classes = list(dist.keys())
    counts = list(dist.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(classes, counts, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("CIFAR-10 Training Set Class Distribution")
    ax.set_ylim(0, max(counts) * 1.15)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Saved class distribution to {save_path}")
    plt.close()


def plot_pixel_intensity_histogram(dataset, n_samples=2000, save_path="plots/pixel_histogram.png"):
    """Histogram of pixel intensities per channel."""
    import os
    os.makedirs("plots", exist_ok=True)

    r_vals, g_vals, b_vals = [], [], []
    for i, (img, _) in enumerate(dataset):
        if i >= n_samples:
            break
        r_vals.extend(img[0].flatten().tolist())
        g_vals.extend(img[1].flatten().tolist())
        b_vals.extend(img[2].flatten().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, vals, color, channel in zip(
        axes,
        [r_vals, g_vals, b_vals],
        ["red", "green", "blue"],
        ["Red", "Green", "Blue"]
    ):
        ax.hist(vals, bins=50, color=color, alpha=0.7, density=True)
        ax.set_title(f"{channel} Channel")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Density" if channel == "Red" else "")

    fig.suptitle("Pixel Intensity Distributions by Channel", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Saved pixel histogram to {save_path}")
    plt.close()


# --- 4. Augmentation pipeline (to be used in training) ---
def build_augmentation_transforms(mean, std):
    """Standard CIFAR-10 augmentation + normalization pipeline."""
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transforms, test_transforms


# --- Main ---
if __name__ == "__main__":
    print("=== CIFAR-10 Data Exploration ===\n")

    # Load raw
    print("Loading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_raw_dataset()
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test  samples: {len(test_dataset)}")
    print(f"  Image shape:   {train_dataset[0][0].shape}  (C x H x W)\n")

    # Class distribution
    print("Computing class distribution...")
    dist = class_distribution(train_dataset)
    for cls, cnt in dist.items():
        print(f"  {cls:<12}: {cnt}")
    print()

    # Channel statistics
    print("Computing channel mean/std (first 5000 samples)...")
    mean, std = compute_channel_stats(train_dataset, num_samples=5000)
    print(f"  Mean : R={mean[0]:.4f}  G={mean[1]:.4f}  B={mean[2]:.4f}")
    print(f"  Std  : R={std[0]:.4f}  G={std[1]:.4f}  B={std[2]:.4f}\n")

    # Augmentation transforms preview
    train_tf, test_tf = build_augmentation_transforms(mean, std)
    print("Augmentation pipeline built:")
    print(f"  Train: {train_tf}")
    print(f"  Test : {test_tf}\n")

    # Visualizations
    print("Generating visualizations...")
    visualize_samples(train_dataset, n_per_class=5)
    plot_class_distribution(dist)
    plot_pixel_intensity_histogram(train_dataset, n_samples=2000)

    print("\nDay 1 complete. Plots saved to ./plots/")
    print("Next: Build CNN architecture from scratch (Day 2)")
