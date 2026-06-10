"""Day 4 - ancestral sampling from the trained DDPM, plus a schedule comparison.

Days 1-3 built the forward process, the U-Net noise predictor, and the training
loop. Now we run the reverse process to actually generate digits. Starting from
pure Gaussian noise x_T, we walk backwards t = T-1 ... 0, at each step using the
network's noise estimate to take one denoising move:

    x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) )
              + sqrt(posterior_var_t) * z,    z ~ N(0, I)  (and z = 0 at t = 0)

This is "ancestral" sampling because each x_{t-1} is drawn from the model's
estimate of the reverse conditional q(x_{t-1} | x_t, x_0). The mean term is the
DDPM reverse mean rewritten in terms of the predicted noise; the added noise z
keeps the chain stochastic until the final step.

The reverse-step coefficients (sqrt(1/alpha_t), the posterior variance, etc.) are
all derived here from the schedule tensors that day 1 already precomputes, so
nothing in day 1 has to change. The script also samples under both the linear and
cosine beta schedules and lays the grids side by side - the cosine schedule
destroys information more slowly and usually yields cleaner late-training digits.
"""

import torch

from day1_forward import DiffusionSchedule
from day2_unet import UNet


def posterior_coefficients(schedule):
    """Derive the reverse-step tensors from the forward schedule.

    Returns the three coefficients each ancestral step needs:
      * sqrt_recip_alphas      = 1 / sqrt(alpha_t)
      * betas / sqrt(1 - alpha_bar_t)  - the predicted-noise scaling
      * posterior_variance     - var of q(x_{t-1} | x_t, x_0), used for z
    """
    alphas = schedule.alphas
    alphas_bar = schedule.alphas_bar
    betas = schedule.betas

    # alpha_bar shifted by one step; alpha_bar_{-1} := 1 so t=0 is well defined
    alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    noise_coef = betas / schedule.sqrt_one_minus_alphas_bar
    # closed-form posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    return sqrt_recip_alphas, noise_coef, posterior_variance


@torch.no_grad()
def p_sample(model, coeffs, x_t, t, t_index):
    """One reverse step: take x_t to x_{t-1} using the predicted noise.

    `t` is the batched timestep tensor fed to the network; `t_index` is the plain
    integer step so we know when to stop adding noise (no z on the final step).
    """
    sqrt_recip_alphas, noise_coef, posterior_variance = coeffs
    a = sqrt_recip_alphas[t_index]
    c = noise_coef[t_index]

    eps = model(x_t, t)
    mean = a * (x_t - c * eps)

    if t_index == 0:
        return mean
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(posterior_variance[t_index]) * noise


@torch.no_grad()
def sample(model, schedule, n_images=16, image_size=28, device="cpu"):
    """Generate a batch of images by running the full reverse chain from noise."""
    model.eval()
    coeffs = tuple(c.to(device) for c in posterior_coefficients(schedule))

    x = torch.randn(n_images, 1, image_size, image_size, device=device)
    for t_index in reversed(range(schedule.timesteps)):
        t = torch.full((n_images,), t_index, device=device, dtype=torch.long)
        x = p_sample(model, coeffs, x, t, t_index)

    # network operates in [-1, 1]; map back to [0, 1] for viewing/saving
    return (x.clamp(-1, 1) + 1) / 2


def compare_schedules(model, timesteps=300, n_images=16, device="cpu"):
    """Sample from the same model under linear and cosine schedules.

    Returns a dict of schedule-name -> image batch. In a real run these batches
    are passed to torchvision.utils.make_grid and saved under samples/; we keep
    plotting out of the import path so the module stays headless-friendly.
    """
    out = {}
    for name in ("linear", "cosine"):
        schedule = DiffusionSchedule(timesteps=timesteps, schedule=name)
        out[name] = sample(model, schedule, n_images=n_images, device=device)
    return out


def load_ema_model(path="ddpm_mnist.pt", device="cpu"):
    """Rebuild the U-Net and load the EMA weights saved by day 3."""
    model = UNet().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["ema"])
    return model


def save_grid(batch, path):
    """Tile a batch into a single grid PNG (optional, needs torchvision)."""
    from torchvision.utils import make_grid, save_image

    grid = make_grid(batch, nrow=int(len(batch) ** 0.5))
    save_image(grid, path)
    print(f"saved sample grid to {path}")


if __name__ == "__main__":
    # smoke test on CPU with an untrained net and a tiny schedule: this only
    # checks the reverse chain runs end to end and returns clean [0, 1] images.
    # Real sampling calls load_ema_model() on the day-3 checkpoint first.
    torch.manual_seed(0)
    model = UNet()

    schedule = DiffusionSchedule(timesteps=20, schedule="linear")
    imgs = sample(model, schedule, n_images=4, device="cpu")
    print(f"sampled batch shape: {tuple(imgs.shape)}")
    print(f"pixel range: [{imgs.min():.3f}, {imgs.max():.3f}]")

    grids = compare_schedules(model, timesteps=20, n_images=4, device="cpu")
    for name, batch in grids.items():
        print(f"{name:6s} schedule -> mean pixel {batch.mean():.3f}")
