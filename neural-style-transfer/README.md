# Neural Style Transfer with PyTorch

Gatys-style neural style transfer using a pretrained VGG19. Content is matched
on a deep conv layer, style is matched via Gram matrices on multiple shallow
layers, and the input image is optimized directly with L-BFGS.

## What's in here

| File | Purpose |
|------|---------|
| `day1_features.py` | VGG19 feature extractor, image load/save pipeline, layer config |
| `day2_losses.py` | Content loss, style loss with Gram matrices, total variation |
| `day3_optimize.py` | L-BFGS optimization loop with snapshots and lr schedule |
| `day4_blending.py` | Multi-style blending, comparison grid for the README |

## Quick start

```bash
pip install torch torchvision pillow
python day3_optimize.py --content inputs/cat.jpg --style inputs/starry.jpg --out outputs/result.png
```

Multi-style blending (weights are normalized automatically):

```bash
python day4_blending.py \
    --content inputs/cat.jpg \
    --blends "inputs/starry.jpg:0.7,inputs/wave.jpg:0.3" \
             "inputs/starry.jpg:0.5,inputs/wave.jpg:0.5" \
    --out-dir outputs/
```

## Notes from running this

- Starting from a clone of the content image converges much faster than starting
  from white noise. Pure-noise init produces interesting but messy first 50
  steps before the content emerges.
- L-BFGS is well-suited here because the input is a tiny number of parameters
  (just the image) but the loss is very nonlinear. Adam works but takes ~5x
  more steps for similar quality.
- The total variation regularizer at ~1e-6 keeps the result from getting too
  speckled without washing out fine detail. At 1e-4 you start losing brush
  strokes; at 0 you get visible high-frequency noise.
- For the blending experiments, evenly weighted styles tend to produce muddy
  results unless the styles share a palette. 70/30 splits look more deliberate.

## References

- Gatys, Ecker, Bethge - "A Neural Algorithm of Artistic Style" (2015)
- Johnson et al. - "Perceptual Losses for Real-Time Style Transfer" (2016)
