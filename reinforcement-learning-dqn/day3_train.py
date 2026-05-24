"""
Day 3: DQN training loop on CartPole-v1.

Pulls together the Q-network and replay buffer from day 2 and adds:

  * epsilon-greedy action selection with linear decay from EPS_START to EPS_END
  * a target network that is hard-synced every TARGET_SYNC steps
  * Huber loss on the temporal-difference error so big errors do not blow up
  * Adam optimiser with gradient clipping (||g||_inf <= 1.0)

The reward curve is logged per episode and dumped to a small csv at the end so
day 4 can compare vanilla DQN with the Double-DQN variant on the same axes.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork, ReplayBuffer


ENV_ID = "CartPole-v1"
SEED = 0

# training schedule
NUM_EPISODES = 400
MAX_STEPS_PER_EP = 500
BUFFER_CAPACITY = 50_000
BATCH_SIZE = 64
WARMUP_STEPS = 1_000
GAMMA = 0.99
LR = 1e-3

# epsilon-greedy schedule
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 10_000

# target network sync cadence in environment steps
TARGET_SYNC = 500

LOG_DIR = Path(__file__).parent / "logs"


def linear_epsilon(step: int) -> float:
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(net: QNetwork, obs: np.ndarray, eps: float, n_actions: int) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = net(torch.from_numpy(obs).unsqueeze(0))
    return int(torch.argmax(q, dim=1).item())


def compute_td_loss(
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
    # huber keeps gradients well-scaled when q estimates are still wild early on
    return nn.functional.smooth_l1_loss(q_pred, q_target)


def train() -> list[float]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNetwork(obs_dim, n_actions)
    target = QNetwork(obs_dim, n_actions)
    target.load_state_dict(online.state_dict())
    target.eval()

    buffer = ReplayBuffer(BUFFER_CAPACITY, obs_dim, seed=SEED)
    optimiser = optim.Adam(online.parameters(), lr=LR)

    total_steps = 0
    episode_returns: list[float] = []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
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
                loss = compute_td_loss(online, target, batch, GAMMA)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                optimiser.step()

            if total_steps % TARGET_SYNC == 0:
                target.load_state_dict(online.state_dict())

            if done:
                break

        episode_returns.append(ep_return)
        if (ep + 1) % 20 == 0:
            window = episode_returns[-20:]
            mean = sum(window) / len(window)
            print(
                f"ep {ep + 1:4d} | steps {total_steps:6d} | "
                f"eps {eps:.3f} | return(20) {mean:6.1f}"
            )

    env.close()
    return episode_returns


def dump_curve(returns: list[float], tag: str = "dqn") -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    out = LOG_DIR / f"{tag}_returns.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return"])
        for i, r in enumerate(returns):
            w.writerow([i, r])
    return out


if __name__ == "__main__":
    returns = train()
    path = dump_curve(returns, tag="dqn")
    print(f"wrote {path}")
    best = max(returns)
    last20 = sum(returns[-20:]) / min(20, len(returns))
    print(f"best episode return : {best:.1f}")
    print(f"mean of last 20 eps : {last20:.1f}")
