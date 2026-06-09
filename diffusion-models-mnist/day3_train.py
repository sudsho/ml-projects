"""Day 3 - the DDPM training loop on the simplified objective, with EMA.

The reverse-process network from day 2 is trained with the surprisingly simple
loss that falls out of the DDPM derivation: sample a clean image x_0, pick a
random timestep t, corrupt it to x_t with known noise eps, and ask the network
to predict eps from (x_t, t). The objective is just

    L_simple = E_{x_0, t, eps} || eps - eps_theta(x_t, t) ||^2

i.e. an MSE between the true and predicted noise. No KL terms, no per-timestep
weighting - Ho et al. found dropping those (the "simplified" loss) trains better.

Two practical pieces live here on top of the bare loss:

  * Uniform timestep sampling - every step we draw t ~ Uniform{0, ..., T-1} per
    image so the network sees the whole noise range, not just easy or hard t.
  * An exponential moving average (EMA) of the weights. Diffusion samples are
    visibly cleaner when drawn from EMA weights rather than the raw optimizer
    weights, because the EMA smooths out the noisy late-training updates. We keep
    a shadow copy and update it after every step.

Day 4 loads the EMA weights and runs ancestral sampling to actually generate
digits.
"""

import copy

import torch
import torch.nn.functional as F

from day1_forward import DiffusionSchedule, load_mnist
from day2_unet import UNet


class EMA:
    """Maintains a shadow copy of the model weights as an exponential moving
    average. decay close to 1 means the shadow moves slowly and stays smooth.

    Usage: construct from a model, call update(model) after each optimizer step,
    and copy_to(model) when you want the averaged weights for sampling."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            # s = decay * s + (1 - decay) * m, done in place
            s_param.mul_(self.decay).add_(m_param, alpha=1 - self.decay)
        # buffers (e.g. GroupNorm has none, but be safe) just track the model
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    @torch.no_grad()
    def copy_to(self, model):
        for m_param, s_param in zip(model.parameters(), self.shadow.parameters()):
            m_param.copy_(s_param)


def p_losses(model, schedule, x_0, t):
    """The simplified DDPM loss for a batch of clean images and timesteps.

    Draw fresh noise, build x_t in closed form from day 1, predict the noise
    back, and return the MSE. This single function is the entire training signal.
    """
    noise = torch.randn_like(x_0)
    x_t = schedule.q_sample(x_0, t, noise=noise)
    predicted = model(x_t, t)
    return F.mse_loss(noise, predicted)


def move_schedule_to(schedule, device):
    """The schedule precomputes coefficient tensors on the CPU; move the ones
    the loss touches onto the training device so gather() lines up."""
    schedule.sqrt_alphas_bar = schedule.sqrt_alphas_bar.to(device)
    schedule.sqrt_one_minus_alphas_bar = schedule.sqrt_one_minus_alphas_bar.to(device)
    return schedule


def train(epochs=5, timesteps=300, lr=2e-4, batch_size=128, device=None):
    """Full training loop. Returns the model, the EMA wrapper, and the schedule
    so day 4 can sample from them."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    schedule = move_schedule_to(DiffusionSchedule(timesteps=timesteps), device)
    loader = load_mnist(batch_size=batch_size)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=0.999)

    history = []
    for epoch in range(epochs):
        running = 0.0
        for step, (x_0, _) in enumerate(loader):
            x_0 = x_0.to(device)
            # one random timestep per image in the batch
            t = torch.randint(0, timesteps, (x_0.size(0),), device=device)

            loss = p_losses(model, schedule, x_0, t)

            optimizer.zero_grad()
            loss.backward()
            # diffusion gradients can spike on near-pure-noise timesteps; clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step:4d}  loss {loss.item():.4f}")

        avg = running / len(loader)
        history.append(avg)
        print(f"== epoch {epoch} done, avg loss {avg:.4f} ==")

    return model, ema, schedule, history


def save_checkpoint(model, ema, path="ddpm_mnist.pt"):
    """Persist both the raw and EMA weights; day 4 samples from the EMA set."""
    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)
    torch.save(
        {"model": model.state_dict(), "ema": ema_model.state_dict()},
        path,
    )
    print(f"saved checkpoint to {path}")


if __name__ == "__main__":
    # a short smoke run so the file is exercisable without a GPU: one tiny
    # synthetic batch through the loss and an EMA step. The real run calls
    # train() with the MNIST loader above.
    torch.manual_seed(0)
    schedule = DiffusionSchedule(timesteps=300)
    model = UNet()
    ema = EMA(model)

    x_0 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 300, (4,))
    loss = p_losses(model, schedule, x_0, t)
    print(f"smoke loss: {loss.item():.4f}")

    before = next(iter(ema.shadow.parameters())).clone()
    # nudge the model so the EMA has something to move toward, then update
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    ema.update(model)
    after = next(iter(ema.shadow.parameters()))
    moved = not torch.allclose(before, after)
    print(f"ema shadow moved after update: {moved}")
