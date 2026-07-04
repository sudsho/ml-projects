"""Day 3 - the combined cross-entropy + soft Dice loss and the training loop.

Days 1-2 gave us image/mask batches at 128x128 and a U-Net that maps an image to
a [B, NUM_CLASSES, H, W] logit map. Today we train it. Two pieces are new:

1. The loss. Plain pixel-wise cross-entropy works, but on this dataset the three
   classes are badly imbalanced - "background" and "foreground" dominate while the
   thin "boundary" ring is a small fraction of the pixels, so a network can score a
   low CE just by ignoring boundaries. Soft Dice loss operates on region overlap
   rather than per-pixel likelihood, which pushes back on that. We optimise the sum
   of the two: CE gives a smooth, well-conditioned gradient early on, Dice keeps the
   minority class from being washed out. Dice is computed on softmax probabilities
   (the "soft" part) so it stays differentiable.

2. The metric. Mean IoU (Jaccard) over the classes is the standard segmentation
   score. We accumulate per-class intersection and union counts across the whole
   validation set and divide once at the end - averaging per-batch IoUs would bias
   the number on batches where a class is absent.

Day 4 turns the trained model into mask overlays and a per-class IoU breakdown.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from day1_data import IMAGE_SIZE, NUM_CLASSES, get_dataloaders
from day2_model import UNet, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_dice_loss(logits, targets, eps=1e-6):
    """Soft (differentiable) multi-class Dice loss.

    logits  : [B, C, H, W] raw scores.
    targets : [B, H, W] integer class labels in [0, C).

    We softmax the logits into probabilities, one-hot the targets to the same
    shape, then compute Dice = 2*|P intersect G| / (|P| + |G|) per class, summed
    over the batch and spatial dims. The loss is 1 - mean Dice over classes.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    true_1hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)  # sum over batch + spatial, keep the class axis
    intersection = torch.sum(probs * true_1hot, dims)
    cardinality = torch.sum(probs + true_1hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy + soft Dice, optionally weighted between the two."""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        dice = soft_dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice, ce.detach(), dice.detach()


class IoUAccumulator:
    """Streaming per-class intersection/union counts for a mean-IoU at the end."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.double)
        self.union = torch.zeros(num_classes, dtype=torch.double)

    def update(self, preds, targets):
        # preds, targets : [B, H, W] integer labels
        for c in range(self.num_classes):
            p = preds == c
            t = targets == c
            self.intersection[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()

    def per_class_iou(self):
        # Classes never present in preds or targets have union 0 -> report nan
        iou = self.intersection / self.union.clamp(min=1)
        iou[self.union == 0] = float("nan")
        return iou

    def mean_iou(self):
        iou = self.per_class_iou()
        valid = iou[~torch.isnan(iou)]
        return valid.mean().item() if valid.numel() else float("nan")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss, _, _ = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running = 0.0
    iou = IoUAccumulator(NUM_CLASSES)
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(images)
        loss, _, _ = criterion(logits, masks)
        running += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        iou.update(preds.cpu(), masks.cpu())
    return running / len(loader.dataset), iou.mean_iou(), iou.per_class_iou()


def fit(epochs=20, lr=1e-3, batch_size=16, weight_decay=1e-4):
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"U-Net trainable params: {count_parameters(model):,}")

    criterion = CEDiceLoss(ce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_miou, per_class = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "unet_best.pt")

        pc = ", ".join(f"{v:.3f}" for v in per_class.tolist())
        print(
            f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| mIoU {val_miou:.4f} | per-class [{pc}]"
        )

    print(f"best val mIoU: {best_miou:.4f} (checkpoint -> unet_best.pt)")
    return model, history


if __name__ == "__main__":
    # Sanity-check the loss and metric on random tensors without downloading data.
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.randint(0, NUM_CLASSES, (2, IMAGE_SIZE, IMAGE_SIZE))

    loss_fn = CEDiceLoss()
    total, ce, dice = loss_fn(logits, targets)
    print(f"combined {total.item():.4f} | ce {ce.item():.4f} | dice {dice.item():.4f}")

    acc = IoUAccumulator(NUM_CLASSES)
    acc.update(logits.argmax(1), targets)
    print(f"random-prediction mIoU (expect ~1/{NUM_CLASSES}): {acc.mean_iou():.4f}")
