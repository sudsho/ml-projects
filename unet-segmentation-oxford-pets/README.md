# U-Net Semantic Segmentation on Oxford-IIIT Pet

A U-Net built from scratch in PyTorch for pixel-wise segmentation of the
Oxford-IIIT Pet dataset. Each pixel is classified into one of three classes from
the dataset's trimap - background, pet (foreground), and the boundary ring around
the animal - so the output is a full-resolution label map rather than a single
image label. The network is the classic symmetric encoder/decoder of Ronneberger
et al. (2015), reproduced here without a segmentation library so every piece is
visible.

## Approach

- **Encoder / decoder with skips**: the encoder halves resolution and doubles
  channels stage by stage (learning *what* at the cost of *where*); the decoder
  upsamples back with transposed convolutions. At each decoder stage the
  matching-resolution encoder feature map is concatenated in - the skip connection
  that hands fine spatial detail straight to the upsampling path and makes crisp
  boundaries possible.
- **Double-conv block**: two 3x3 convolutions, each with batch norm and ReLU,
  padded to preserve spatial size so skip tensors line up exactly.
- **Combined loss**: cross-entropy plus a soft (differentiable) Dice term. The
  three classes are badly imbalanced - the boundary ring is a thin minority - and
  plain CE can score low by ignoring it; Dice operates on region overlap and keeps
  the minority class from being washed out.
- **Metric**: mean IoU (Jaccard), accumulated as streaming per-class
  intersection/union counts over the whole validation set and divided once at the
  end, so batches missing a class don't bias the score.

## Files

| File | Day | Contents |
|------|-----|----------|
| `day1_data.py` | 1 | Oxford-IIIT Pet pipeline - joint image/mask transforms, trimap {1,2,3} -> class {0,1,2} remap, resize/normalize, train/val split |
| `day2_model.py` | 2 | U-Net from scratch - double-conv block, downsampling encoder, upsampling decoder with skip-connection concatenation |
| `day3_train.py` | 3 | Combined cross-entropy + soft Dice loss and the Adam training loop with a streaming per-epoch mean-IoU |
| `day4_eval.py` | 4 | Per-class IoU breakdown, qualitative mask overlays, and loss/IoU training curves |

## Why joint augmentation matters

An image transform and its mask transform have to move together - a random crop or
flip applied to the image but not the label map silently destroys supervision.
Day 1 applies geometric transforms jointly to the pair and reserves photometric
ones (normalization) for the image only, since colour changes must not touch the
integer class ids.

## Reading the results

`day4` prints IoU per class rather than only the mean, because the mean hides the
story: background and pet segment well while the boundary class is consistently the
hardest, and that gap is the honest measure of the model. The overlay grid -
image, ground truth, prediction, and an alpha-blended prediction - shows *where*
errors land, which are almost always along the boundary ring that barely moves the
IoU but is obvious to the eye.

## Tech stack

PyTorch, NumPy, Matplotlib.
