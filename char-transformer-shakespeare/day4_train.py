"""Day 4 - train the character transformer and sample from it.

Days 1-3 built the data pipeline, masked self-attention, and the full
decoder-only model. Today we actually train it and read text back out:

  1. an AdamW optimizer with a warmup + cosine learning-rate schedule,
  2. a training loop that periodically estimates train/val loss on held-out
     batches (so the printed numbers are not just noisy single-batch values),
  3. autoregressive sampling - feed the model its own output one character at a
     time, cropping the context to the last `block_size` tokens,
  4. a loss curve saved to disk and a short generated sample written out.

Defaults are sized to train in a few minutes on CPU and far faster on a GPU;
the point is a clean end-to-end run, not a state-of-the-art Shakespeare bot.
"""

import math
import os

import torch

from day1_data import build_pipeline
from day3_model import CharTransformer

# --- hyperparameters ------------------------------------------------------
BLOCK_SIZE = 128
BATCH_SIZE = 64
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1

MAX_ITERS = 5000
WARMUP_ITERS = 100
LR_MAX = 3e-4
LR_MIN = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_ITERS = 200

HERE = os.path.dirname(__file__)
SAMPLES_DIR = os.path.join(HERE, "samples")


def lr_at(step):
    """Linear warmup for WARMUP_ITERS steps, then cosine decay to LR_MIN.

    Warmup keeps the first few updates small while the Adam moment estimates are
    still cold; the cosine tail anneals the rate smoothly toward LR_MIN so the
    end of training takes small, stable steps instead of stopping abruptly.
    """
    if step < WARMUP_ITERS:
        return LR_MAX * (step + 1) / WARMUP_ITERS
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LR_MAX - LR_MIN) * cosine


@torch.no_grad()
def estimate_loss(model, datasets, generator):
    """Average loss over EVAL_ITERS random batches from each split.

    A single batch's loss is too noisy to track progress by, so we average a
    fixed number of batches with the model in eval mode (dropout off) and the
    grads disabled. Returns a dict like {"train": ..., "val": ...}.
    """
    out = {}
    model.eval()
    for name, ds in datasets.items():
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            xb, yb = ds.get_batch(BATCH_SIZE, generator=generator)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(model, vocab, max_new_tokens=500, prompt="\n", temperature=1.0,
             device="cpu", generator=None):
    """Autoregressively sample characters from the trained model.

    Start from `prompt`, and at each step take the logits for the final
    position, divide by `temperature` (lower = greedier, higher = more random),
    softmax, and sample one character. The running context is cropped to the
    last BLOCK_SIZE tokens because the positional embedding only covers that
    many positions.
    """
    model.eval()
    idx = vocab.encode(prompt).unsqueeze(0).to(device)  # (1, T)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return vocab.decode(idx[0])


def save_loss_curve(history, path):
    """Plot train/val loss vs. step, skipping cleanly if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping loss curve")
        return
    steps = [h["step"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [h["train"] for h in history], label="train")
    plt.plot(steps, [h["val"] for h in history], label="val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("char transformer on tiny-shakespeare")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"saved loss curve -> {path}")


def train():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device="cpu").manual_seed(0)

    vocab, train_ds, val_ds = build_pipeline(
        block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, device=device
    )
    datasets = {"train": train_ds, "val": val_ds}

    model = CharTransformer(
        vocab.size, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
    ).to(device)
    print(f"device       : {device}")
    print(f"vocab size   : {vocab.size}")
    print(f"param count  : {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99)
    )

    history = []
    for step in range(MAX_ITERS):
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = train_ds.get_batch(BATCH_SIZE, generator=gen)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            stats = estimate_loss(model, datasets, gen)
            stats["step"] = step
            history.append(stats)
            print(
                f"step {step:5d} | lr {lr:.2e} | "
                f"train {stats['train']:.4f} | val {stats['val']:.4f}"
            )

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    save_loss_curve(history, os.path.join(SAMPLES_DIR, "loss_curve.png"))

    sample = generate(model, vocab, max_new_tokens=1000, device=device, generator=gen)
    sample_path = os.path.join(SAMPLES_DIR, "sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"saved sample -> {sample_path}")
    print("\n----- sample -----")
    print(sample[:500])

    return model, history


if __name__ == "__main__":
    train()
