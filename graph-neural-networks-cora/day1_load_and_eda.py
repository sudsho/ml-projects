"""
Day 1 - Cora citation network loading and EDA.

Cora is the canonical small graph benchmark: 2708 nodes (papers), 5429 edges
(citations), 7 classes, 1433 bag-of-words features per node. We load the
public Planetoid split and look at degree distribution, class balance, and
whether neighbors tend to share a label (homophily) - that last one is what
makes a GCN useful here in the first place.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

try:
    from torch_geometric.datasets import Planetoid
except ImportError:
    Planetoid = None


def load_cora(root="./data"):
    if Planetoid is None:
        raise RuntimeError("install torch-geometric to load Cora")
    dataset = Planetoid(root=root, name="Cora")
    return dataset[0], dataset.num_classes


def degree_stats(edge_index, num_nodes):
    deg = np.zeros(num_nodes, dtype=np.int64)
    src = edge_index[0].cpu().numpy()
    for s in src:
        deg[s] += 1
    return deg


def class_balance(y):
    counts = Counter(y.cpu().numpy().tolist())
    total = sum(counts.values())
    return {k: (v, v / total) for k, v in sorted(counts.items())}


def homophily(edge_index, y):
    # fraction of edges whose endpoints share a label - a sanity check that
    # message passing will actually help on this graph
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    labels = y.cpu().numpy()
    same = (labels[src] == labels[dst]).sum()
    return same / len(src)


def main():
    data, num_classes = load_cora()
    print(f"nodes={data.num_nodes} edges={data.num_edges} feats={data.num_features} classes={num_classes}")

    deg = degree_stats(data.edge_index, data.num_nodes)
    print(f"degree: mean={deg.mean():.2f} median={np.median(deg):.0f} max={deg.max()}")

    print("class balance:")
    for cls, (n, frac) in class_balance(data.y).items():
        print(f"  class {cls}: {n} ({frac:.1%})")

    h = homophily(data.edge_index, data.y)
    print(f"edge homophily: {h:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deg, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("node degree")
    ax.set_ylabel("count")
    ax.set_title("Cora degree distribution")
    fig.tight_layout()
    fig.savefig("cora_degree_hist.png", dpi=120)


if __name__ == "__main__":
    main()
