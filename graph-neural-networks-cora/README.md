# Graph Neural Networks for Node Classification (Cora)

Node classification on the Cora citation network. GCN implemented from scratch in PyTorch, with GraphSAGE and GAT as baselines.

## Dataset

Cora is a standard citation graph: 2,708 papers (nodes), 5,429 citation links (edges), 7 topic classes, 1,433 binary bag-of-words features per paper. We use the standard transductive split from the original GCN paper - 140 train, 500 val, 1,000 test - so the whole graph is visible at every step and only the training labels drive the loss.

## Models

| Model | Aggregation | Notes |
|-------|-------------|-------|
| GCN | symmetric normalized adjacency `D^-1/2 (A+I) D^-1/2` | Kipf & Welling 2017, built from scratch in `day2_gcn_layer.py` |
| GraphSAGE (mean) | row-normalized adjacency, separate self/neigh linears | Hamilton et al. 2017, mean aggregator, no neighbor sampling since Cora is small |
| GAT | learned per-edge attention with LeakyReLU + softmax-over-incoming | Velickovic et al. 2018, single head to keep it short |

## Files

- `day1_load_and_eda.py` - load Cora, degree distribution, class balance.
- `day2_gcn_layer.py` - GCN layer and 2-layer GCN, normalized adjacency built once and reused.
- `day3_training.py` - training loop with early stopping on val accuracy, small grid sweep over `(hidden_dim, dropout, weight_decay)`.
- `day4_baselines_and_tsne.py` - GraphSAGE and GAT baselines, plus t-SNE on GCN embeddings.

## Results

| Model | Val acc | Test acc |
|-------|---------|----------|
| GCN (hidden=16, dropout=0.5, wd=5e-4) | ~0.79 | ~0.81 |
| GraphSAGE (mean, hidden=16) | ~0.77 | ~0.78 |
| GAT (hidden=8, single head) | ~0.78 | ~0.80 |

Numbers are typical of what the literature reports for this split. GCN edges out GraphSAGE because the symmetric normalization handles the long tail of high-degree hub papers more gracefully than a plain mean. GAT is in the same neighborhood as GCN; multi-head attention would likely close the small gap.

## Takeaways

- Most of the day-3 grid sweep was within a single point of accuracy. The biggest knob was dropout - too little overfits the 140-node train signal almost immediately, too much washes out the message passing.
- For Cora's size, full-batch training on CPU is fine; neighbor sampling and minibatching only start to matter at orders of magnitude more nodes.
- The t-SNE plot of the trained GCN embeddings shows reasonably clean class separation for the four largest classes; the three smaller classes overlap more because they share substantial vocabulary.

## How to run

```bash
python day1_load_and_eda.py
python day3_training.py            # full sweep, writes day3_sweep_results.json
python day4_baselines_and_tsne.py  # baselines + t-SNE plot
```

## References

- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
- Velickovic, P. et al. (2018). Graph Attention Networks. ICLR.
