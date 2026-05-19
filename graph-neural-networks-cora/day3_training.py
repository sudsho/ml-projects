"""
Day 3 - GCN training loop on Cora with dropout and weight decay tuning.

Cora is the standard split from the GCN paper: 140 train / 500 val / 1000 test
nodes, all 2708 nodes share the same graph and we get to use the full feature
matrix at every step (this is transductive learning). The training signal only
comes from labels on the 140 train nodes, but messages still propagate over
the whole graph.

We sweep a small grid over (hidden_dim, dropout, weight_decay) and pick the
configuration that maximizes validation accuracy. Then we report test accuracy
on that single picked configuration so we are not selecting on test.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from day1_load_and_eda import load_cora
from day2_gcn_layer import GCN, normalize_adjacency


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    correct = (pred == labels[mask]).float().sum().item()
    return correct / mask.sum().item()


def train_once(features, labels, adj, masks, hp, epochs=200, patience=30, seed=0):
    """Train one model with the given hyperparameters, return best val/test acc."""
    torch.manual_seed(seed)
    train_mask, val_mask, test_mask = masks

    model = GCN(
        in_dim=features.size(1),
        hidden_dim=hp["hidden_dim"],
        num_classes=int(labels.max().item()) + 1,
        dropout=hp["dropout"],
    ).to(DEVICE)

    opt = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )

    best_val = 0.0
    best_test = 0.0
    bad = 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            val_acc = accuracy(logits, labels, val_mask)
            test_acc = accuracy(logits, labels, test_mask)

        # early stopping on val accuracy
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_val, best_test


def main():
    print(f"device: {DEVICE}")
    data = load_cora()
    features = data["features"].to(DEVICE)
    labels = data["labels"].to(DEVICE)
    edge_index = data["edge_index"].to(DEVICE)
    masks = tuple(m.to(DEVICE) for m in (data["train_mask"], data["val_mask"], data["test_mask"]))

    adj = normalize_adjacency(edge_index, features.size(0))

    # small grid - keep this cheap, full sweep would explode
    grid = []
    for hidden in (16, 32, 64):
        for dropout in (0.3, 0.5, 0.7):
            for wd in (1e-4, 5e-4, 1e-3):
                grid.append({"hidden_dim": hidden, "dropout": dropout, "weight_decay": wd, "lr": 0.01})

    results = []
    t0 = time.time()
    for i, hp in enumerate(grid, 1):
        val, test = train_once(features, labels, adj, masks, hp)
        results.append({**hp, "val_acc": val, "test_acc": test})
        print(f"[{i:>2}/{len(grid)}] {hp}  val={val:.4f}  test={test:.4f}")

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    best = results[0]
    print(f"\nbest by val: {best}")
    print(f"sweep took {elapsed:.1f}s over {len(grid)} configs")

    out = Path(__file__).parent / "day3_sweep_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
