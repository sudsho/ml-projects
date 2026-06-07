"""Day 1 - the forward diffusion process for a DDPM on MNIST.

The forward process gradually corrupts a clean image x_0 into pure Gaussian
noise over T timesteps. Because each step adds Gaussian noise with a fixed
variance schedule, we can sample x_t directly from x_0 in closed form without
iterating:

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

This file sets up the noise schedule, implements the closed-form sampler, and
visualizes a digit being progressively destroyed. Days 2-4 add the U-Net,
the training loop, and ancestral sampling.
"""

import torch
import torchvision
from torchvision import transforms


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Original DDPM schedule: betas spaced linearly from start to end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Improved-DDPM cosine schedule. Tends to destroy information more slowly
    early on, which helps sample quality. Returns clipped betas."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputes everything the forward (and later reverse) process needs."""

    def __init__(self, timesteps=300, schedule="linear"):
        self.timesteps = timesteps
        if schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps)
        else:
            self.betas = linear_beta_schedule(timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        # coefficients used by the closed-form q(x_t | x_0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

    def _gather(self, values, t, x_shape):
        """Pick the per-sample coefficient for timestep t and reshape it so it
        broadcasts against a batch of images."""
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """Sample x_t from x_0 in one shot using the closed form above."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self._gather(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_ab = self._gather(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise


def load_mnist(batch_size=128):
    """MNIST scaled to [-1, 1] so it matches the Gaussian noise range."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=tf
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def demo_progressive_noising(schedule, image, steps_to_show=(0, 50, 100, 200, 299)):
    """Return a list of (timestep, noised_image) pairs for visualization.

    A real run would hand these to matplotlib; we keep the plotting optional so
    the module imports cleanly in a headless environment.
    """
    frames = []
    for t in steps_to_show:
        t_tensor = torch.tensor([t])
        x_t = schedule.q_sample(image.unsqueeze(0), t_tensor)
        frames.append((t, x_t.squeeze(0)))
    return frames


if __name__ == "__main__":
    schedule = DiffusionSchedule(timesteps=300, schedule="linear")
    print(f"betas range: {schedule.betas[0]:.5f} -> {schedule.betas[-1]:.5f}")
    print(f"alpha_bar at t=0:   {schedule.alphas_bar[0]:.4f} (almost clean)")
    print(f"alpha_bar at t=299: {schedule.alphas_bar[-1]:.4f} (almost pure noise)")

    # sanity check the closed form on a single fake image
    fake = torch.randn(1, 28, 28)
    frames = demo_progressive_noising(schedule, fake)
    for t, frame in frames:
        print(f"t={t:3d}  mean={frame.mean():+.3f}  std={frame.std():.3f}")
