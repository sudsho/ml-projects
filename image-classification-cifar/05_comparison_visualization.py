# Day 5: Model comparison, confusion matrix, and Grad-CAM visualization
# Compare CNN from scratch vs ResNet transfer learning on CIFAR-10 test set.
# Produces confusion matrices, per-class accuracy, and Grad-CAM heatmaps.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report

# ── reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = 10
BATCH_SIZE = 128


# ── data ───────────────────────────────────────────────────────────────────────
def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── model definitions (must match training scripts) ────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ── plotting helpers ───────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, ax):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.tick_params(axis="x", rotation=45)


def plot_per_class_accuracy(preds, labels, ax, title):
    accs = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        accs.append((preds[mask] == c).mean() * 100)
    bars = ax.barh(CLASSES, accs, color=plt.cm.RdYlGn(np.array(accs) / 100))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=9)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    loader = get_test_loader()

    # ── load or simulate models ────────────────────────────────────────────────
    cnn_model = SimpleCNN().to(DEVICE)
    resnet_model = build_resnet_model().to(DEVICE)

    cnn_path = "cnn_scratch.pth"
    resnet_path = "resnet_finetuned.pth"

    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        print("Loaded CNN from scratch weights.")
    else:
        print("No saved CNN weights found — using random weights for demo.")

    if os.path.exists(resnet_path):
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
        print("Loaded ResNet fine-tuned weights.")
    else:
        print("No saved ResNet weights found — using random weights for demo.")

    # ── evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating CNN from scratch...")
    cnn_acc, cnn_preds, true_labels = evaluate_model(cnn_model, loader)
    print(f"CNN Accuracy: {cnn_acc:.4f} ({cnn_acc*100:.2f}%)")

    print("Evaluating ResNet transfer learning...")
    res_acc, res_preds, _ = evaluate_model(resnet_model, loader)
    print(f"ResNet Accuracy: {res_acc:.4f} ({res_acc*100:.2f}%)")

    # classification reports
    print("\n=== CNN from Scratch ===")
    print(classification_report(true_labels, cnn_preds, target_names=CLASSES))
    print("=== ResNet Transfer Learning ===")
    print(classification_report(true_labels, res_preds, target_names=CLASSES))

    # ── confusion matrices ─────────────────────────────────────────────────────
    cnn_cm = confusion_matrix(true_labels, cnn_preds)
    res_cm = confusion_matrix(true_labels, res_preds)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_confusion_matrix(cnn_cm, "CNN from Scratch\n(Normalized)", axes[0])
    plot_confusion_matrix(res_cm, "ResNet Transfer Learning\n(Normalized)", axes[1])
    plt.suptitle("CIFAR-10 Confusion Matrices — Model Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrices.png")

    # ── per-class accuracy comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_per_class_accuracy(cnn_preds, true_labels, axes[0], "CNN from Scratch — Per-Class Accuracy")
    plot_per_class_accuracy(res_preds, true_labels, axes[1], "ResNet Transfer Learning — Per-Class Accuracy")
    plt.suptitle("Per-Class Accuracy Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved per_class_accuracy.png")

    # ── accuracy bar comparison ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    model_names = ["CNN from Scratch", "ResNet18 (Transfer)"]
    accuracies = [cnn_acc * 100, res_acc * 100]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(model_names, accuracies, color=colors, width=0.4, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("CIFAR-10 — CNN vs Transfer Learning", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_comparison.png")

    # ── Grad-CAM on sample images ──────────────────────────────────────────────
    test_set = datasets.CIFAR10(root="./data", train=False, download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010)),
                                ]))

    # get one sample per class (first occurrence)
    class_samples = {}
    for img, label in test_set:
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) == NUM_CLASSES:
            break

    # Grad-CAM on ResNet (last conv layer)
    target_layer = resnet_model.layer4[-1].conv2
    grad_cam = GradCAM(resnet_model, target_layer)

    inv_normalize = transforms.Normalize(
        mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
        std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010],
    )

    fig, axes = plt.subplots(3, NUM_CLASSES, figsize=(20, 6))
    for c in range(NUM_CLASSES):
        img_tensor = class_samples[c]
        cam, pred_idx = grad_cam.generate(img_tensor)

        # original image (denormalized)
        raw = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        raw = np.clip(raw, 0, 1)

        # overlay
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * raw + 0.5 * heatmap

        axes[0, c].imshow(raw)
        axes[0, c].set_title(CLASSES[c], fontsize=9, fontweight="bold")
        axes[0, c].axis("off")

        axes[1, c].imshow(cam, cmap="jet")
        axes[1, c].axis("off")

        axes[2, c].imshow(np.clip(overlay, 0, 1))
        axes[2, c].set_xlabel(f"pred: {CLASSES[pred_idx]}", fontsize=8)
        axes[2, c].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9, rotation=0, labelpad=50)
    axes[2, 0].set_ylabel("Overlay", fontsize=9, rotation=0, labelpad=50)
    fig.suptitle("Grad-CAM Visualizations — ResNet Transfer Learning", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("gradcam_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved gradcam_visualization.png")

    print("\n=== Final Summary ===")
    print(f"CNN from Scratch:          {cnn_acc*100:.2f}%")
    print(f"ResNet Transfer Learning:  {res_acc*100:.2f}%")
    improvement = res_acc - cnn_acc
    print(f"Transfer learning gain:    {improvement*100:+.2f}%")


if __name__ == "__main__":
    main()
