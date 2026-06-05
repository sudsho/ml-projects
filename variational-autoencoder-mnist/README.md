# Variational Autoencoder on MNIST

A VAE built from scratch in PyTorch on MNIST. The project goes from a plain
(non-variational) autoencoder baseline to a full VAE with the reparameterization
trick and the ELBO loss, then explores the reconstruction-vs-KL tradeoff and
finally probes what the latent space actually learned.

## Layout

| Day | File | What it does |
|-----|------|--------------|
| 1 | `day1_data_and_baseline.py` | MNIST dataloaders and a vanilla (deterministic) autoencoder as a reconstruction-quality baseline |
| 2 | `day2_vae_elbo.py` | VAE encoder/decoder, reparameterization trick, ELBO loss (BCE + KL), training loop |
| 3 | `day3_beta_and_annealing.py` | beta-VAE sweep and KL annealing schedule, reconstruction-vs-KL tradeoff plot |
| 4 | `day4_latent_viz.py` | latent traversals, sampling from the prior, t-SNE of encoded means |

## Key ideas

- **Reparameterization trick.** Sampling `z = mu + sigma * eps`, `eps ~ N(0, I)`
  keeps the sampling step differentiable w.r.t. the encoder parameters, so the
  KL and reconstruction gradients flow back through it.
- **ELBO.** Loss is a reconstruction term (Bernoulli BCE over pixels) plus the
  closed-form KL between the approximate posterior `q(z|x)` and the standard
  normal prior. Both are kept per-sample so the numbers stay comparable to the
  day 1 baseline.
- **beta / KL annealing.** Scaling the KL term (beta-VAE) trades reconstruction
  for a more prior-like latent; ramping beta up over the first few epochs lets
  the decoder start using `z` before the KL clamps the posterior, which helps
  avoid posterior collapse.

## What the latent space looks like (day 4)

- **Traversals.** Walking the most active latent dims (ranked by the variance of
  the encoded means) morphs a digit smoothly - slant, stroke thickness, and
  loop size shift continuously rather than jumping between classes.
- **Prior samples.** Decoding `z ~ N(0, I)` produces plausible (if slightly
  blurry) digits, confirming the posterior was pulled toward the prior.
- **t-SNE.** The encoded means cluster by digit even though the VAE is trained
  fully unsupervised and never sees the labels.

## Running

```bash
python day1_data_and_baseline.py   # baseline AE
python day2_vae_elbo.py            # train the VAE
python day3_beta_and_annealing.py  # beta sweep + tradeoff plot
python day4_latent_viz.py          # latent traversals, prior samples, t-SNE
```

MNIST downloads automatically on first run. Figures land in `./plots/`.

## Stack

PyTorch, torchvision, matplotlib, scikit-learn (t-SNE).
