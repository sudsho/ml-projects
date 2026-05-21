"""
Day 4 - GraphSAGE and GAT baselines on Cora, plus a t-SNE plot of the
learned node embeddings of the best GCN configuration from day 3.

GCN aggregates with a fixed normalized adjacency. The baselines try two
different aggregation schemes:

  - GraphSAGE (mean): each node averages its neighbors' features and then
    concatenates with its own, followed by a linear and nonlinearity. No
    normalized adjacency, neighbor sampling is left full for Cora since the
    graph is small.

  - GAT: learns per-edge attention weights via a small linear+LeakyReLU
    scoring network, then weights neighbor features by softmax over the
    incoming edges. Single head here to keep this short.

We run each with the same train/val/test split, report test accuracy at the
epoch with best val accuracy, and finally plot t-SNE on the penultimate
embeddings of the picked GCN config.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from day1_load_and_eda import load_cora
from day2_gcn_layer import GCN, normalize_adjacency
from day3_training import accuracy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, adj):
        # adj is the row-normalized adjacency (so adj @ x is the mean of neighbors)
        neigh = torch.sparse.mm(adj, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.l1 = SAGELayer(in_dim, hidden_dim)
        self.l2 = SAGELayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.l1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l2(h, adj)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, leaky=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # additive attention vector split into source and target halves
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)
        self.leaky = nn.LeakyReLU(leaky)

    def forward(self, x, edge_index):
        h = self.W(x)
        src, dst = edge_index[0], edge_index[1]
        # raw attention score per edge
        e = self.leaky((h[src] * self.a_src).sum(-1) + (h[dst] * self.a_dst).sum(-1))
        # softmax over edges grouped by destination
        e = e - e.max()
        e_exp = e.exp()
        denom = torch.zeros(h.size(0), device=h.device).scatter_add_(0, dst, e_exp) + 1e-16
        alpha = e_exp / denom[dst]
        # weighted sum of source features into each dst
        out = torch.zeros_like(h)
        out.index_add_(0, dst, alpha.unsqueeze(1) * h[src])
        return out


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.6):
        super().__init__()
        self.l1 = GATLayer(in_dim, hidden_dim)
        self.l2 = GATLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.elu(self.l1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l2(h, edge_index)


def row_normalize_adjacency(edge_index, num_nodes):
    """Row-normalized A + I as a sparse COO tensor. Used by GraphSAGE for mean aggregation."""
    self_loops = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    w = 1.0 / deg[row]
    return torch.sparse_coo_tensor(edge_index, w, (num_nodes, num_nodes)).coalesce()


def add_self_loops(edge_index, num_nodes):
    self_loops = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, self_loops], dim=1)


def train_model(model, forward_kwargs, features, labels, masks, lr=0.005, wd=5e-4, epochs=200, patience=30):
    train_mask, val_mask, test_mask = masks
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_test, bad = 0.0, 0.0, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(features, **forward_kwargs)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(features, **forward_kwargs)
            val_acc = accuracy(logits, labels, val_mask)
            test_acc = accuracy(logits, labels, test_mask)
        if val_acc > best_val:
            best_val, best_test, bad = val_acc, test_acc, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_val, best_test


def tsne_plot(embeddings, labels, out_path):
    """Save a 2D t-SNE scatter of node embeddings colored by class."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn/matplotlib not available, skipping t-SNE plot")
        return
    proj = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, s=6, cmap="tab10")
    plt.title("GCN embeddings (t-SNE, Cora)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"wrote {out_path.name}")


def main():
    print(f"device: {DEVICE}")
    data = load_cora()
    features = data["features"].to(DEVICE)
    labels = data["labels"].to(DEVICE)
    edge_index = data["edge_index"].to(DEVICE)
    masks = tuple(m.to(DEVICE) for m in (data["train_mask"], data["val_mask"], data["test_mask"]))
    num_nodes = features.size(0)
    num_classes = int(labels.max().item()) + 1

    sym_adj = normalize_adjacency(edge_index, num_nodes)
    row_adj = row_normalize_adjacency(edge_index, num_nodes)
    ei_self = add_self_loops(edge_index, num_nodes)

    summary = {}

    # GCN baseline at a single reasonable config (matches the best from day 3 sweep)
    gcn = GCN(in_dim=features.size(1), hidden_dim=16, num_classes=num_classes, dropout=0.5).to(DEVICE)
    val, test = train_model(gcn, {"adj": sym_adj}, features, labels, masks, lr=0.01, wd=5e-4)
    summary["GCN"] = {"val": val, "test": test}
    print(f"GCN     val={val:.4f}  test={test:.4f}")

    sage = GraphSAGE(in_dim=features.size(1), hidden_dim=16, num_classes=num_classes).to(DEVICE)
    val, test = train_model(sage, {"adj": row_adj}, features, labels, masks)
    summary["GraphSAGE"] = {"val": val, "test": test}
    print(f"SAGE    val={val:.4f}  test={test:.4f}")

    gat = GAT(in_dim=features.size(1), hidden_dim=8, num_classes=num_classes).to(DEVICE)
    val, test = train_model(gat, {"edge_index": ei_self}, features, labels, masks, lr=0.005, wd=5e-4)
    summary["GAT"] = {"val": val, "test": test}
    print(f"GAT     val={val:.4f}  test={test:.4f}")

    out = Path(__file__).parent / "day4_baseline_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out.name}")

    # t-SNE of the GCN's penultimate embeddings on all nodes
    gcn.eval()
    with torch.no_grad():
        # pull representation right before the final classification layer
        h = gcn.gc1(features, sym_adj)
        h = F.relu(h)
    tsne_plot(h.cpu().numpy(), labels.cpu().numpy(), Path(__file__).parent / "day4_tsne.png")


if __name__ == "__main__":
    main()
