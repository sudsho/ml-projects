"""Day 4 - qualitative overlays, per-class IoU, and training curves.

Days 1-3 built the data pipeline, the U-Net, and the CE + Dice training loop with
a streaming mean-IoU. Day 4 is the read-out: turn the trained checkpoint into
things a human can actually judge.

Three deliverables:

  1. Per-class IoU breakdown. The mean IoU from day 3 hides the story - on this
     dataset "background" and "pet" score well while the thin "boundary" ring is
     always the hardest class. We re-run the streaming accumulator over the
     validation set and print IoU per class, not just the mean, so the weak spot
     is visible.

  2. Qualitative mask overlays. Metrics don't show *where* the model is wrong.
     We render a small grid: input image, ground-truth mask, predicted mask, and
     the prediction alpha-blended over the image. Boundary errors that barely move
     the IoU are obvious to the eye here.

  3. Training curves. Train/val loss and val mIoU per epoch from day 3's history,
     saved as a figure - the standard sanity check for over/underfitting.

The heavy lifting (a real checkpoint, real Oxford-IIIT Pet data) comes from day 3;
everything here is written to also run end to end on random tensors so the plotting
and metric code can be smoke-tested on CPU without a download. Guarded imports keep
the module usable when matplotlib is absent.
"""

import numpy as np
import torch

from day1_data import IMAGE_SIZE, NUM_CLASSES, get_dataloaders
from day2_model import UNet
from day3_train import DEVICE, IoUAccumulator

# Trimap class ids after the day-1 {1,2,3} -> {0,1,2} remap.
CLASS_NAMES = ["background", "pet", "boundary"]

# A fixed RGB colour per class for mask rendering (background/pet/boundary).
CLASS_COLORS = np.array(
    [
        [0.15, 0.15, 0.15],  # background - near black
        [0.15, 0.55, 0.95],  # pet        - blue
        [0.95, 0.75, 0.10],  # boundary   - amber
    ],
    dtype=np.float32,
)


def load_model(checkpoint="unet_best.pt"):
    """Rebuild the U-Net and load a day-3 checkpoint if one exists.

    Falls back to a randomly initialised model so the visualisation code path is
    exercisable without having trained first - the overlays are then meaningless
    but the shapes and plotting are validated.
    """
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    try:
        state = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"loaded checkpoint: {checkpoint}")
    except FileNotFoundError:
        print(f"no checkpoint at {checkpoint} - using randomly initialised weights")
    model.eval()
    return model


@torch.no_grad()
def per_class_iou(model, loader):
    """Stream the validation set once and return (per_class_iou, mean_iou)."""
    acc = IoUAccumulator(NUM_CLASSES)
    for images, masks in loader:
        preds = model(images.to(DEVICE)).argmax(dim=1)
        acc.update(preds.cpu(), masks)
    return acc.per_class_iou(), acc.mean_iou()


def report_iou(per_class, mean):
    """Pretty-print the per-class IoU table and the mean underneath."""
    print("\nper-class IoU")
    print("-" * 28)
    for name, iou in zip(CLASS_NAMES, per_class.tolist()):
        shown = "  n/a" if iou != iou else f"{iou:.3f}"  # iou != iou catches nan
        print(f"  {name:<10} {shown}")
    print("-" * 28)
    print(f"  {'mean':<10} {mean:.3f}\n")


def colorize(mask):
    """Map an [H, W] integer label map to an [H, W, 3] RGB image via CLASS_COLORS."""
    return CLASS_COLORS[mask]


@torch.no_grad()
def save_overlays(model, loader, n=4, path="overlays.png"):
    """Save an image / truth / prediction / blended-overlay grid for n samples."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping overlay figure")
        return

    images, masks = next(iter(loader))
    images, masks = images[:n], masks[:n]
    preds = model(images.to(DEVICE)).argmax(dim=1).cpu()

    # Undo the day-1 ImageNet normalisation for display.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    shown = (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    cols = ["image", "ground truth", "prediction", "overlay"]
    fig, axes = plt.subplots(n, 4, figsize=(11, 3 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        pred_rgb = colorize(preds[r].numpy())
        blended = 0.55 * shown[r] + 0.45 * pred_rgb
        panels = [shown[r], colorize(masks[r].numpy()), pred_rgb, blended]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def save_curves(history, path="training_curves.png"):
    """Plot train/val loss and val mIoU vs epoch from day 3's history dict."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping curve figure")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("CE + Dice loss")
    ax_loss.set_title("loss")
    ax_loss.legend()

    ax_iou.plot(epochs, history["val_miou"], color="tab:green")
    ax_iou.set_xlabel("epoch")
    ax_iou.set_ylabel("mean IoU")
    ax_iou.set_title("validation mIoU")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    """Full read-out on real data - expects a day-3 checkpoint and the dataset."""
    _, val_loader = get_dataloaders(batch_size=16)
    model = load_model()
    per_class, mean = per_class_iou(model, val_loader)
    report_iou(per_class, mean)
    save_overlays(model, val_loader)


if __name__ == "__main__":
    # Smoke test on random tensors: no dataset, no checkpoint. Validates that the
    # IoU report, colorizer, and curve plotting all run and produce sane shapes.
    torch.manual_seed(0)

    fake_preds = torch.randint(0, NUM_CLASSES, (8, IMAGE_SIZE, IMAGE_SIZE))
    fake_truth = torch.randint(0, NUM_CLASSES, (8, IMAGE_SIZE, IMAGE_SIZE))
    acc = IoUAccumulator(NUM_CLASSES)
    acc.update(fake_preds, fake_truth)
    report_iou(acc.per_class_iou(), acc.mean_iou())

    rgb = colorize(fake_preds[0].numpy())
    assert rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), rgb.shape
    print(f"colorized mask shape: {rgb.shape}")

    fake_history = {
        "train_loss": [1.2, 0.8, 0.6, 0.5],
        "val_loss": [1.3, 0.9, 0.72, 0.68],
        "val_miou": [0.31, 0.48, 0.55, 0.58],
    }
    save_curves(fake_history, path="training_curves.png")
    print("day 4 smoke test done")
