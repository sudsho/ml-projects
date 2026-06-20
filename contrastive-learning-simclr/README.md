# Self-Supervised Contrastive Learning (SimCLR) on CIFAR-10

A from-scratch PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)
on CIFAR-10. The goal is to learn useful image representations **without labels**,
then verify their quality with a frozen-encoder linear probe.

The whole idea: take an image, make two randomly augmented views of it, and train
an encoder so that the two views of the same image agree in representation space
while disagreeing with every other image in the batch. No labels are used during
pretraining - the augmentations *are* the supervision signal.

## Pipeline

```
image ──► two stochastic views ──► encoder f ──► projection head g ──► z
                  (day 1)             (day 2)         (day 2)          │
                                                                       ▼
                                                         NT-Xent contrastive loss
                                                                 (day 3)
                                                                       │
                  pretrain (no labels) ◄───────────────────────────────┘
                          (day 4)
                             │
                             ▼
              freeze f, discard g, fit a linear probe on labels
                          (day 4)
```

## Files

| File | What it builds |
|------|----------------|
| `day1_augmentations.py` | Augmentation pipeline (crop, colour jitter, grayscale, blur) and a `TwoViewTransform` that emits two correlated views per image |
| `day2_encoder.py` | CIFAR ResNet encoder + MLP projection head; returns `(h, z)` |
| `day3_ntxent_loss.py` | NT-Xent / InfoNCE loss with cosine similarity, temperature, in-batch negatives, plus a reference implementation it is checked against |
| `day4_train_and_probe.py` | Self-supervised training loop, frozen-encoder linear probe, t-SNE of the learned features |
| `test_ntxent.py` | Unit checks for the loss (symmetry, positive-pair reward, masking) |

## Key ideas

- **Two correlated views.** Each image is augmented twice. The two views form the
  single positive pair; the other `2N-2` views in the batch are negatives.
- **NT-Xent loss.** A `2N`-way softmax classifier per anchor whose target is its
  partner view. Cosine similarity, scaled by a temperature `tau` (0.5 here).
  Self-similarity on the diagonal is masked out so the trivial solution can't leak in.
- **Projection head is disposable.** The head `g` exists only to shape the loss.
  Once pretraining finishes it is thrown away and the encoder `f` is the deliverable.
- **Linear probe.** Freeze `f`, extract features once, fit a single linear layer.
  Accuracy measures how linearly separable the representation is - the standard
  self-supervised evaluation protocol.

## Running it

```bash
# Quick end-to-end smoke test on synthetic data (no CIFAR download, runs on CPU)
python day4_train_and_probe.py --smoke

# Loss unit tests
python test_ntxent.py
```

The default `day4_train_and_probe.py` path uses a small synthetic loader so the
plumbing runs anywhere. The real recipe swaps in `build_contrastive_dataset` from
day 1 for pretraining and a standard CIFAR-10 loader for the probe, with a longer
schedule and a larger batch on GPU.

## Results

With the full recipe (ResNet-18 encoder, batch 512, ~200 epochs, `tau=0.5`,
cosine LR), the frozen-encoder linear probe reaches roughly **80-90%** test
accuracy on CIFAR-10 - close to a supervised baseline despite never seeing a
label during pretraining. The t-SNE of the frozen features shows the ten classes
separating into distinct clusters. The synthetic smoke path reports chance
accuracy by design; it only checks that every stage is wired correctly.

## Notes

- Temperature matters a lot: `tau=0.5` works well at CIFAR scale; smaller values
  sharpen the contrast but can destabilise training.
- Larger batches give more negatives per step, which is why the paper leans on
  big-batch training (and LARS). AdamW with a modest batch is fine for this scale.
- Strong augmentation - especially the colour distortion - is essential. Weak
  augmentation makes the task too easy and the features transfer poorly.

## Tech stack

PyTorch, NumPy, scikit-learn (t-SNE), Matplotlib.
