"""Day 4 - put it all together: pretrain SimCLR, then evaluate the features.

The previous three days built every piece in isolation:

  - day 1: a stochastic augmentation pipeline that turns one image into two
    correlated views (crop, colour jitter, grayscale, optional blur).
  - day 2: a CIFAR ResNet encoder f plus an MLP projection head g, returning
    (h, z) where h is the representation we ultimately care about.
  - day 3: the NT-Xent loss that pulls partner views together and pushes the
    other 2N-2 views in the batch apart.

Today wires them into a real run. Two phases:

  1. Self-supervised pretraining. No labels at all - we optimise NT-Xent over
     batches of two-view pairs. The projection head g is only a "loss adaptor";
     once training is done we throw it away and keep the encoder f.

  2. Linear probe. The standard way to score self-supervised features: freeze
     the encoder, extract h for every (labelled) image once, and fit a single
     linear classifier on top. Accuracy here measures how linearly separable the
     learned representation is - if pretraining worked, a linear layer is enough.

We also run t-SNE on a sample of the frozen features to eyeball whether the ten
CIFAR classes have pulled apart in representation space.

The defaults below are deliberately small so the file runs on a laptop CPU for a
smoke test; the real recipe (longer training, larger batch, GPU) is noted inline
and in the project README. Pass --smoke to force the tiny synthetic path.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from day2_encoder import SimCLRModel, count_parameters
from day3_ntxent_loss import NTXentLoss


def pretrain(model, loader, *, epochs, lr, temperature, weight_decay, device):
    """Self-supervised SimCLR pretraining with NT-Xent.

    `loader` is expected to yield batches of (view1, view2), each of shape
    (N, 3, H, W) - the two augmented views produced by day 1's TwoViewTransform.
    Returns the per-epoch mean loss so we can plot a curve later.
    """
    model.to(device).train()
    criterion = NTXentLoss(temperature=temperature)

    # LARS is the paper's optimiser for big batches; AdamW is plenty for the
    # small-batch CPU regime here and avoids the extra dependency.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        running, batches = 0.0, 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)

            # Encode both views, stack so row i and row i+N are a positive pair,
            # exactly the layout day 3's loss expects.
            _, z1 = model(view1)
            _, z2 = model(view2)
            z = torch.cat([z1, z2], dim=0)

            loss = criterion(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            batches += 1

        scheduler.step()
        mean_loss = running / max(batches, 1)
        history.append(mean_loss)
        print(f"epoch {epoch + 1:3d}/{epochs}  ntxent={mean_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.5f}")
    return history


@torch.no_grad()
def extract_features(encoder, loader, device):
    """Run the frozen encoder over a labelled loader and collect (h, y).

    This is done once before fitting the probe - there is no reason to re-run the
    convnet every probe epoch when its weights never change.
    """
    encoder.to(device).eval()
    feats, labels = [], []
    for images, y in loader:
        h = encoder(images.to(device))
        feats.append(h.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(train_feats, train_labels, test_feats, test_labels,
                 *, num_classes, epochs, lr, device):
    """Fit a single linear layer on frozen features and report test accuracy.

    The encoder stays frozen by construction: we never touch it here, only the
    Linear(h_dim -> num_classes) on top of pre-extracted features.
    """
    in_dim = train_feats.shape[1]
    clf = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=0.0)

    # Standardise features - linear probes are sensitive to feature scale and
    # this consistently helps. Statistics come from the train split only.
    mean, std = train_feats.mean(0, keepdim=True), train_feats.std(0, keepdim=True) + 1e-6
    train_x = ((train_feats - mean) / std).to(device)
    test_x = ((test_feats - mean) / std).to(device)
    train_y, test_y = train_labels.to(device), test_labels.to(device)

    clf.train()
    for epoch in range(epochs):
        logits = clf(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(epochs // 5, 1) == 0:
            print(f"  probe epoch {epoch + 1:3d}/{epochs}  ce={loss.item():.4f}")

    clf.eval()
    with torch.no_grad():
        preds = clf(test_x).argmax(dim=1)
        acc = (preds == test_y).float().mean().item()
    return acc


def tsne_embedding(feats, labels, *, max_points=1000, seed=0):
    """2-D t-SNE of a feature sample, returned as (xy, labels) for plotting.

    Falls back gracefully if scikit-learn is unavailable so the smoke test still
    completes; the project README shows the actual scatter.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn not installed - skipping t-SNE (see README for the plot)")
        return None, None

    if feats.shape[0] > max_points:
        idx = torch.randperm(feats.shape[0])[:max_points]
        feats, labels = feats[idx], labels[idx]

    xy = TSNE(n_components=2, init="pca", perplexity=30,
              random_state=seed).fit_transform(feats.numpy())
    return xy, labels.numpy()


def _synthetic_views(n=64, image_size=16, batch_size=16):
    """A tiny two-view loader so the file runs end-to-end without downloading
    CIFAR-10. Each 'image' is noise; this only checks the plumbing, not accuracy."""
    base = torch.randn(n, 3, image_size, image_size)
    v1 = base + 0.1 * torch.randn_like(base)
    v2 = base + 0.1 * torch.randn_like(base)
    return DataLoader(TensorDataset(v1, v2), batch_size=batch_size, shuffle=True)


def _synthetic_labelled(n=128, image_size=16, num_classes=10, batch_size=32):
    images = torch.randn(n, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="SimCLR pretrain + linear probe")
    parser.add_argument("--smoke", action="store_true",
                        help="run the tiny synthetic path (no CIFAR download)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"device: {device}")

    # The smoke path uses synthetic noise so CI / a quick local run never blocks
    # on the CIFAR-10 download. The real run swaps these for day 1's
    # build_contrastive_dataset (pretrain) and a standard CIFAR-10 loader (probe).
    pre_loader = _synthetic_views()
    train_loader = _synthetic_labelled(n=256)
    test_loader = _synthetic_labelled(n=128)

    model = SimCLRModel()
    print(f"SimCLR params: {count_parameters(model):,}")

    print("\n[1/3] self-supervised pretraining")
    pretrain(model, pre_loader, epochs=args.epochs, lr=args.lr,
             temperature=args.temperature, weight_decay=args.weight_decay,
             device=device)

    print("\n[2/3] linear probe on frozen features")
    train_feats, train_labels = extract_features(model.encoder, train_loader, device)
    test_feats, test_labels = extract_features(model.encoder, test_loader, device)
    acc = linear_probe(train_feats, train_labels, test_feats, test_labels,
                       num_classes=10, epochs=args.probe_epochs, lr=1e-2, device=device)
    # On real CIFAR-10 with the full recipe this lands around 0.80-0.90; on the
    # synthetic smoke path it is chance (~0.10) by design.
    print(f"linear-probe test accuracy: {acc:.4f}")

    print("\n[3/3] t-SNE of frozen features")
    xy, lab = tsne_embedding(train_feats, train_labels)
    if xy is not None:
        print(f"t-SNE embedding ready: {xy.shape[0]} points in 2-D")

    print("\ndone - encoder is the deliverable; projection head is discarded.")


if __name__ == "__main__":
    main()
