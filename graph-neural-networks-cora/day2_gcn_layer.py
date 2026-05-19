"""
Day 2 - Build a GCN layer from scratch in PyTorch.

The propagation rule from Kipf and Welling (2017) is:

    H' = sigma( D^-1/2 (A + I) D^-1/2  H  W )

Adding the identity gives self-loops so each node aggregates itself, the
symmetric normalization D^-1/2 (.) D^-1/2 keeps activations from blowing up
on high-degree nodes, and W is the learned linear transform. We compute the
normalized adjacency once (it is constant for a given graph) and reuse it
every forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(edge_index, num_nodes):
    """Return D^-1/2 (A + I) D^-1/2 as a sparse COO tensor."""
    # add self loops by appending [i, i] for every i
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loops = self_loops.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    # symmetric normalization weight for each edge: 1 / sqrt(deg_row * deg_col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        edge_index, weights, (num_nodes, num_nodes)
    ).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        # glorot - common for GCN
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        # linear transform then propagate via normalized adjacency
        x = x @ self.weight
        out = torch.sparse.mm(adj_norm, x)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """Two-layer GCN, the model used in the original paper for Cora."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = F.relu(self.gc1(x, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gc2(h, adj_norm)


def smoke_test():
    """Tiny sanity check - 4 nodes in a path graph, 3 input features, 2 classes."""
    torch.manual_seed(0)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    n = 4
    x = torch.randn(n, 3)
    adj = normalize_adjacency(edge_index, n)

    model = GCN(in_dim=3, hidden_dim=8, num_classes=2)
    logits = model(x, adj)
    assert logits.shape == (n, 2), logits.shape
    print("smoke test ok, logits =\n", logits.detach())


if __name__ == "__main__":
    smoke_test()
