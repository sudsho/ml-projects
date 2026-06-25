# Deep Convolutional GAN (DCGAN) on Fashion-MNIST

A Deep Convolutional GAN built from scratch in PyTorch that generates 28x28
Fashion-MNIST images. The project follows the DCGAN architectural recipe (Radford
et al., 2015): a transposed-convolution generator, a strided-convolution
discriminator, and the alternating adversarial training loop, then sampling and
latent-space interpolation to inspect what the generator learned.

Unlike the VAE and diffusion projects in this repo, a GAN has no explicit
likelihood and no reconstruction term. The generator turns a noise vector into an
image, the discriminator tries to separate real images from generated ones, and
the only learning signal is the discriminator's verdict.

## Approach

The DCGAN paper is essentially a set of architectural rules that make the
adversarial game stable enough to train with plain convolutions:

- strided convolutions instead of pooling in the discriminator, fractionally
  strided (transposed) convolutions in the generator, so the nets learn their own
  spatial down/up-sampling
- batch norm in both networks, except the generator output and discriminator input
  layers
- ReLU in the generator with a Tanh output, LeakyReLU in the discriminator
- no fully connected hidden layers

Fashion-MNIST is single channel at 28x28. To keep the transposed-conv arithmetic
clean, the generator produces 32x32 and the reals are resized 28 -> 32 to match,
so every spatial dimension is a power of two (4 -> 8 -> 16 -> 32).

## Files

| File | Day | Contents |
|------|-----|----------|
| `day1_models.py` | 1 | Fashion-MNIST data pipeline normalised to [-1, 1], generator and discriminator architectures, DCGAN weight init |
| `day2_losses.py` | 2 | BCEWithLogits adversarial losses, real/fake label convention with one-sided label smoothing, the alternating G/D step |
| `day3_train.py` | 3 | Full training loop, separate Adam optimisers (beta1=0.5), fixed-noise sample grids per epoch, mode-collapse monitoring |
| `day4_sampling.py` | 4 | Final sample grid, slerp latent-space interpolation, generator/discriminator loss curves |

## Training recipe

- **Loss**: `BCEWithLogitsLoss` on the discriminator's raw logits (no final
  sigmoid), with the non-saturating generator loss (relabel fakes as real)
- **Labels**: real = 0.9 (one-sided label smoothing), fake = 0.0
- **Optimisers**: two independent Adam, lr 2e-4, betas (0.5, 0.999) - the lowered
  beta1 damps the back-and-forth oscillation of the adversarial game
- **Monitoring**: D(x) and D(G(z)) before/after each G update; healthy training
  keeps both loosely around 0.5 with a visible gap, rather than D(x) -> 1

## Latent-space interpolation

Interpolation between two noise vectors uses spherical linear interpolation
(slerp) rather than a straight line. The Gaussian prior concentrates its mass in a
thin shell at radius ~sqrt(latent_dim); a straight lerp dips through the
sparsely-populated interior near the origin where the generator was never trained
and emits blurry averages. Slerp stays on the shell the whole way, so every
intermediate latent is as typical as the endpoints and the decoded walk morphs one
garment smoothly into another.

## Usage

Each day's module runs its own self-contained sanity check:

```bash
python day1_models.py     # shape checks, parameter counts
python day2_losses.py     # one synthetic step, confirm both nets move
python day3_train.py      # short CPU-capped training run
python day4_sampling.py   # tiny train run, then grids + interpolation + curves
```

For a real run, call `train()` from `day3_train.py` with `max_steps_per_epoch=None`
over several epochs, then pass the returned dict to `export_artifacts()` in
`day4_sampling.py` to write the final sample grid, the latent interpolation, and
the loss curves to `samples/`.

## Requirements

PyTorch, torchvision, NumPy, and Matplotlib. A GPU helps but the project runs on
CPU; Fashion-MNIST downloads automatically on first use.
