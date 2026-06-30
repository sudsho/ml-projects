# Vision Transformer (ViT) from Scratch on CIFAR-10

A Vision Transformer built from scratch in PyTorch that classifies CIFAR-10. The
project follows the original ViT recipe (Dosovitskiy et al., 2020): chop the image
into a grid of fixed-size patches, treat each patch as a token, prepend a learnable
class token, add learnable positional embeddings, and run the sequence through a
standard transformer encoder - no convolutions in the model itself. The encoder's
final state at the class-token position is read by a linear head.

The repo already has a decoder-only transformer for text
(`char-transformer-shakespeare`); a ViT is the mirror image of that idea applied to
images. The provocative claim of the paper is that once images are tokenized,
the *same* attention machinery used for language can classify them.

## Approach

The model is a deliberately plain stack so the architecture, not a bag of tricks,
is what you read:

- **Patch embedding**: a `Conv2d` with `kernel = stride = patch_size` splits the
  32x32 image into a grid of patches and projects each to the model width in one
  fused op (a 4x4 patch -> a 192-dim token, giving an 8x8 = 64-token grid).
- **Class token + positional embeddings**: one learnable CLS vector is prepended;
  learnable positional embeddings (CLS included) restore the order that
  permutation-invariant attention would otherwise lose.
- **Encoder blocks**: pre-norm transformer blocks - multi-head self-attention and
  an MLP, each wrapped in a residual connection.
- **Head**: a final LayerNorm, then a linear layer reading only the CLS token.

## Files

| File | Day | Contents |
|------|-----|----------|
| `day1_patch_embed.py` | 1 | CIFAR-10 pipeline with augmentation, patch embedding (conv-as-linear), class token, learnable positional embeddings |
| `day2_attention.py` | 2 | Scaled dot-product attention, multi-head self-attention, the pre-norm encoder block (attention + MLP, residuals) |
| `day3_vit_model.py` | 3 | Full ViT assembly and the training loop - AdamW, cosine LR with warmup, label smoothing, gradient clipping |
| `day4_eval_attention.py` | 4 | Test-set evaluation with per-class accuracy, CLS attention-map visualization, a small-CNN baseline comparison |

## Training recipe

- **Optimizer**: AdamW (lr 3e-4, weight decay 0.05). With no convolutional
  inductive bias, a from-scratch ViT leans on decoupled weight decay.
- **Schedule**: short linear warmup then cosine decay - attention logits are
  unstable early, and warmup avoids an opening divergence.
- **Regularization**: label smoothing (0.1) and gradient clipping (max-norm 1.0),
  both cheap and reliable on a model that would otherwise overfit 50k images.

## Looking inside the model

Self-attention is one of the few deep architectures you can inspect directly.
`day4` pulls the attention weights from the final encoder block, takes the CLS
row (how much the class token attends to each patch), reshapes it to the patch
grid, and upsamples to an image-sized heatmap - so you can see which regions the
classifier's summary token actually drew from. The single-block map extends to
full attention rollout (Abnar & Zuidema, 2020) by multiplying the per-layer maps.

## Baseline

A compact three-stage CNN of comparable parameter count is trained on the same
data. On a dataset as small as CIFAR-10 the convolutional inductive bias is a real
advantage, so the CNN is a fair and genuinely strong point of comparison rather
than a straw man.

## Tech stack

PyTorch, NumPy, Matplotlib.
