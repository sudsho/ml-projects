"""
Day 4: Double-DQN variant and head-to-head comparison with vanilla DQN.

Vanilla DQN forms its td target with

    y = r + gamma * max_a' Q_target(s', a')

which uses the same network both to pick the argmax action and to evaluate it.
That double use of Q_target tends to overestimate values because any noisy
positive error is preferentially selected by the max.

Double-DQN decouples the two:

    a*  = argmax_a' Q_online(s', a')
    y   = r + gamma * Q_target(s', a*)

The online net picks the action, the target net evaluates it. Same compute,
same architecture, only the loss changes. We reuse the day-3 training loop and
toggle the td-loss with a flag, then plot both curves on one figure so the
difference is visible across seeds.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork, ReplayBuffer
from day3_train import (
    BATCH_SIZE,
    BUFFER_CAPACITY,
    ENV_ID,
    EPS_DECAY_STEPS,
    EPS_END,
    EPS_START,
    GAMMA,
    LR,
    MAX_STEPS_PER_EP,
    NUM_EPISODES,
    TARGET_SYNC,
    WARMUP_STEPS,
    linear_epsilon,
    select_action,
)


LOG_DIR = Path(__file__).parent / "logs"


def double_dqn_loss(
    online: QNetwork,
    target: QNetwork,
    batch: tuple,
    gamma: float,
) -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch
    q_pred = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=1, keepdim=True)
        q_next = target(next_states).gather(1, next_actions).squeeze(1)
        q_target = rewards + gamma * q_next * (1.0 - dones)
    return nn.functional.smooth_l1_loss(q_pred, q_target)


def vanilla_dqn_loss(
    online: QNetwork,
    target: QNetwork,
    batch: tuple,
    gamma: float,
) -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch
    q_pred = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target(next_states).max(dim=1).values
        q_target = rewards + gamma * q_next * (1.0 - dones)
    return nn.functional.smooth_l1_loss(q_pred, q_target)


def train_one(seed: int, double: bool) -> list[float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNetwork(obs_dim, n_actions)
    target = QNetwork(obs_dim, n_actions)
    target.load_state_dict(online.state_dict())
    target.eval()

    buffer = ReplayBuffer(BUFFER_CAPACITY, obs_dim, seed=seed)
    optimiser = optim.Adam(online.parameters(), lr=LR)
    loss_fn = double_dqn_loss if double else vanilla_dqn_loss

    total_steps = 0
    returns: list[float] = []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        for _ in range(MAX_STEPS_PER_EP):
            eps = linear_epsilon(total_steps)
            action = select_action(online, obs, eps, n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, float(reward), next_obs, terminated)

            obs = next_obs
            ep_return += float(reward)
            total_steps += 1

            if len(buffer) >= max(BATCH_SIZE, WARMUP_STEPS):
                batch = buffer.sample(BATCH_SIZE)
                loss = loss_fn(online, target, batch, GAMMA)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                optimiser.step()

            if total_steps % TARGET_SYNC == 0:
                target.load_state_dict(online.state_dict())

            if done:
                break
        returns.append(ep_return)

    env.close()
    return returns


def smoothed(xs: list[float], window: int = 20) -> list[float]:
    if window <= 1 or len(xs) < window:
        return list(xs)
    out: list[float] = []
    s = sum(xs[:window])
    out.extend([s / window] * window)
    for i in range(window, len(xs)):
        s += xs[i] - xs[i - window]
        out.append(s / window)
    return out


def dump(returns: list[float], tag: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{tag}_returns.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return"])
        for i, r in enumerate(returns):
            w.writerow([i, r])
    return path


def comparison_plot(vanilla: list[float], double: list[float]) -> Path:
    import matplotlib.pyplot as plt

    LOG_DIR.mkdir(exist_ok=True)
    out = LOG_DIR / "dqn_vs_double_dqn.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(smoothed(vanilla), label="DQN (smoothed)")
    ax.plot(smoothed(double), label="Double DQN (smoothed)")
    ax.axhline(475, linestyle="--", linewidth=0.8, label="solved threshold")
    ax.set_xlabel("episode")
    ax.set_ylabel("return (20-ep moving avg)")
    ax.set_title("CartPole-v1: DQN vs Double DQN")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    vanilla = train_one(seed=args.seed, double=False)
    double = train_one(seed=args.seed, double=True)
    dump(vanilla, "dqn")
    dump(double, "double_dqn")

    last20_v = sum(vanilla[-20:]) / 20
    last20_d = sum(double[-20:]) / 20
    print(f"final 20-ep mean | dqn: {last20_v:6.1f} | double dqn: {last20_d:6.1f}")

    if not args.no_plot:
        path = comparison_plot(vanilla, double)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
