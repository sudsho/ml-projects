# RealNVP Normalizing Flow on MNIST

A RealNVP normalizing flow written from scratch in PyTorch. It maps MNIST digits
to a standard Gaussian through a stack of affine coupling layers, and because
every layer is invertible with a cheap log-determinant, the model reports the
*exact* log-likelihood of the data instead of a bound on it. Training is plain
maximum likelihood, sampling is one inverse pass, and encoding a real image
returns the latent that reproduces it bit for bit.

## Idea

A flow models a complicated density by pushing a simple one through an
invertible map. If `z = f(x)` and `f` is a diffeomorphism, the change of
variables formula gives the density exactly:

    log p_x(x) = log p_z(f(x)) + log |det df/dx|

The whole design problem is making that determinant cheap. An affine coupling
layer solves it by splitting the vector with a binary mask: the masked half is
copied through untouched, and it alone conditions the scale and translation
applied to the other half.

    y_masked   = x_masked
    y_unmasked = x_unmasked * exp(s(x_masked)) + t(x_masked)

The Jacobian is triangular, so its log-determinant is just `sum(s)` no matter how
large or nonlinear the networks producing `s` and `t` are, and inverting is
subtraction and division rather than a solve. A single layer leaves half the
coordinates alone, so the layers are stacked with the mask parity flipped between
them, and the log-determinants add along the composition.

Two preprocessing steps come before any of that, and both are load-bearing.
Discrete pixel values would let a continuous density put infinite mass on the 256
allowed levels, so uniform dequantization noise is added first. And the flow
lives on all of R^D while pixels live in [0, 1], so a logit transform opens the
interval up to the real line. Both contribute their own log-determinant term.

## Layout

- `day1_coupling_layer.py` - preprocessing (dequantization, logit transform with
  its log-det) and one `AffineCoupling` layer with its masked scale/translate
  network, plus the checkerboard mask. The scale output goes through a
  `tanh`-and-scale parameterization to keep `exp(s)` from exploding early.
- `day2_flow_model.py` - `RealNVP`, the stack with alternating masks, and the
  exact likelihood: base Gaussian log-density plus accumulated log-determinants,
  converted to bits per dimension. Asserts round-trip invertibility and
  log-determinant consistency, which is the only way to know the flow is a flow.
- `day3_train.py` - the maximum-likelihood loop. Fresh dequantization noise every
  epoch, gradient-norm clipping with a per-epoch clip count, cosine learning-rate
  decay, held-out bpd, and a sample grid per epoch.
- `day4_sampling.py` - temperature-controlled sampling, slerp vs lerp
  interpolation, interpolation between two encoded real digits, and the bpd
  curves (`--plot`).

## Key design choices

- **Bits per dimension, not raw nats.** `-log p(x) / (D ln 2) + 8` is the
  comparable number across density models on 8-bit images, and the `+8` undoes
  the `/256` from dequantization. Skipping it makes the model look better than it
  is.
- **Dequantization noise resampled every epoch.** The noise is part of the data
  distribution being modelled, not a one-time preprocessing artifact. Caching one
  noisy copy per image lets the model memorize the noise and quietly reports a
  bpd below the honest value.
- **Gradient clipping plus a bounded scale.** The log-determinant term is
  unbounded below, so the loss can always be lowered by driving `s` very
  negative. The `tanh` bound on the scale and a clipped global gradient norm are
  the two halves of the defence against that turning into a NaN.
- **Spherical interpolation between latents.** In 784 dimensions a standard
  Gaussian concentrates on a shell of radius ~28. The straight line between two
  typical latents sags to norm ~21 at its midpoint, off the shell and into a
  region with almost no training mass, which is exactly where interpolations go
  washed out. slerp holds the norm near 29 the whole way across. Both are
  generated so the difference is visible rather than claimed.
- **Temperature is a viewing knob, not a result.** Scaling the base draw by
  `T < 1` cleans up samples, but those are no longer draws from the model, so
  the temperature grid is never used as evidence about likelihood.

## Results

Ten epochs on MNIST with 6 coupling layers and 256-unit scale/translate nets:

| quantity                          | value      |
|-----------------------------------|------------|
| held-out bits/dim, untrained      | ~13        |
| held-out bits/dim, 10 epochs      | ~1.4       |
| encode/decode round-trip error    | ~1e-5      |

The round-trip number is the one worth dwelling on. A trained flow reproduces an
input image to floating-point precision, so the "reconstructions" in the
interpolation grid are the actual dataset digits. The comparison against the VAE
in this repo is the clean one: same dataset, same latent dimension, but the VAE
optimizes a lower bound and its encoder throws information away, so it cannot
report a true likelihood and cannot reconstruct exactly. What a flow pays for
that is architectural freedom - every layer has to stay invertible with a
tractable determinant, which is why it needs depth and width where a VAE decoder
gets to be an arbitrary network.

Samples stop improving well before bpd stops falling, which is the standard
reminder that likelihood and perceptual quality are different axes.

## Run

```bash
python day1_coupling_layer.py   # coupling layer + preprocessing checks
python day2_flow_model.py       # stacked flow, invertibility, log-det checks
python day3_train.py            # maximum-likelihood training in bits/dim
python day4_sampling.py --plot  # samples, interpolations, bpd curves
```

MNIST downloads to `data/` on first run. Sample grids and plots are written to
`samples/` and are gitignored.
