"""
Day 2: Q-network architecture and replay buffer.

Two pieces here that day 3 will glue together with a training loop:

  * QNetwork - a small MLP that maps observation -> action-value vector.
    CartPole obs is 4-dim and we only have 2 actions, so even a tiny net
    learns this quickly. We keep the depth/width configurable so we can
    swap it out later without touching training code.

  * ReplayBuffer - a fixed-size circular buffer of transitions plus a
    uniform random sampler. Stores everything as numpy and only converts
    to tensors at sample time, which keeps the per-step cost low.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"],
)


class QNetwork(nn.Module):
    """MLP Q-function: state -> Q(s, a) for every discrete action."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Iterable[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity circular replay buffer with uniform sampling."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._pos
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = float(done)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if batch_size > self._size:
            raise ValueError(
                f"asked for {batch_size} samples but only have {self._size}"
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._states[idx]),
            torch.from_numpy(self._actions[idx]),
            torch.from_numpy(self._rewards[idx]),
            torch.from_numpy(self._next_states[idx]),
            torch.from_numpy(self._dones[idx]),
        )


def smoke_test() -> None:
    """Tiny end-to-end check: build net, push transitions, sample, forward."""
    torch.manual_seed(0)
    net = QNetwork(obs_dim=4, n_actions=2)
    buf = ReplayBuffer(capacity=1000, obs_dim=4, seed=0)

    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.standard_normal(4).astype(np.float32)
        a = int(rng.integers(0, 2))
        r = float(rng.standard_normal())
        ns = rng.standard_normal(4).astype(np.float32)
        d = bool(rng.integers(0, 2))
        buf.push(s, a, r, ns, d)

    states, actions, rewards, next_states, dones = buf.sample(32)
    q = net(states)
    print(f"buffer size       : {len(buf)}")
    print(f"sampled states    : {tuple(states.shape)}")
    print(f"sampled actions   : {tuple(actions.shape)}")
    print(f"q-values shape    : {tuple(q.shape)}")
    print(f"q-value mean/std  : {q.mean().item():.4f} / {q.std().item():.4f}")


if __name__ == "__main__":
    smoke_test()
