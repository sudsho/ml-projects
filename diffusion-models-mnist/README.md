# Denoising Diffusion Model on MNIST

A DDPM (Ho et al., 2020) built from scratch in PyTorch on MNIST: the forward
noising process, a small U-Net noise predictor, training on the simplified loss,
and ancestral sampling - plus a linear-vs-cosine schedule comparison.

## Overview

A diffusion model learns to reverse a fixed noising process. The forward process
gradually turns a clean digit into Gaussian noise over `T` steps; a network is
trained to predict the noise added at each step, and sampling runs that network
backwards from pure noise to generate new digits.

## Layout

| Day | File | What it covers |
|-----|------|----------------|
| 1 | `day1_forward.py` | Forward process - linear & cosine beta schedules, closed-form `q(x_t \| x_0)`, progressive-noising demo |
| 2 | `day2_unet.py` | U-Net noise predictor with sinusoidal timestep embeddings and residual blocks |
| 3 | `day3_train.py` | Training loop on the simplified DDPM loss, gradient clipping, EMA of weights |
| 4 | `day4_sample.py` | Ancestral sampling, linear-vs-cosine schedule comparison, sample grids |

## Method notes

- **Forward process.** `q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`
  lets us jump straight to any timestep in closed form, so training never has to
  simulate the chain step by step.
- **Objective.** The simplified loss is a plain MSE between the true noise and
  the network's prediction, `|| eps - eps_theta(x_t, t) ||^2`. Dropping the KL
  weighting trains more stably than the full variational bound.
- **EMA.** Samples are drawn from an exponential moving average of the weights,
  which smooths out late-training jitter and visibly cleans up the digits.
- **Sampling.** The reverse step rewrites the DDPM posterior mean in terms of the
  predicted noise and adds posterior-variance noise at every step except the last.
- **Schedules.** The cosine schedule destroys information more slowly early on
  than the linear one, which tends to improve sample quality.

## Running

```bash
python day1_forward.py    # inspect the noise schedule
python day2_unet.py       # parameter count and a shape sanity check
python day3_train.py      # train; saves ddpm_mnist.pt (raw + EMA weights)
python day4_sample.py     # sample from the EMA weights, save grids to samples/
```

Each file also has a `__main__` smoke test that runs on CPU without a trained
checkpoint, so the modules can be exercised in isolation.

## Stack

PyTorch, torchvision. Model artifacts (`*.pt`), the MNIST download (`data/`), and
generated grids (`samples/`, `*.png`) are kept out of git.
