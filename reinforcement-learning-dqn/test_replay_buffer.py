"""
Unit tests for the replay buffer. Light test set, just enough to catch
the obvious mistakes - wrong shapes, off-by-one on the circular index,
sampling more than we have.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q_network import QNetwork, ReplayBuffer


def _fake_transition(obs_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(obs_dim).astype(np.float32)
    ns = rng.standard_normal(obs_dim).astype(np.float32)
    return s, int(rng.integers(0, 2)), float(rng.standard_normal()), ns, False


def test_buffer_starts_empty() -> None:
    buf = ReplayBuffer(capacity=10, obs_dim=4)
    assert len(buf) == 0


def test_buffer_grows_then_caps_at_capacity() -> None:
    cap = 5
    buf = ReplayBuffer(capacity=cap, obs_dim=4)
    for i in range(cap + 3):
        buf.push(*_fake_transition(4, seed=i))
    assert len(buf) == cap


def test_buffer_overwrites_oldest_entry() -> None:
    buf = ReplayBuffer(capacity=3, obs_dim=2)
    # Push four transitions; the first one should have been overwritten.
    for i in range(4):
        state = np.array([float(i), float(i)], dtype=np.float32)
        next_state = np.array([float(i + 1), float(i + 1)], dtype=np.float32)
        buf.push(state, i % 2, float(i), next_state, False)
    # Internal state shouldn't contain the value 0.0 (the first state) anymore.
    assert not np.any(buf._states[:, 0] == 0.0)


def test_sample_returns_correct_shapes_and_types() -> None:
    buf = ReplayBuffer(capacity=64, obs_dim=4)
    for i in range(50):
        buf.push(*_fake_transition(4, seed=i))
    states, actions, rewards, next_states, dones = buf.sample(16)
    assert states.shape == (16, 4)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, 4)
    assert dones.shape == (16,)
    assert states.dtype == torch.float32
    assert actions.dtype == torch.int64


def test_sample_more_than_available_raises() -> None:
    buf = ReplayBuffer(capacity=64, obs_dim=4)
    for i in range(5):
        buf.push(*_fake_transition(4, seed=i))
    with pytest.raises(ValueError):
        buf.sample(10)


def test_qnetwork_output_shape() -> None:
    net = QNetwork(obs_dim=4, n_actions=2)
    x = torch.randn(8, 4)
    q = net(x)
    assert q.shape == (8, 2)


def test_qnetwork_respects_hidden_sizes() -> None:
    net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 16, 8))
    # 3 hidden Linear + 1 output Linear = 4 Linear layers; ReLUs in between.
    linear_layers = [m for m in net.net if hasattr(m, "weight")]
    assert len(linear_layers) == 4
    assert linear_layers[0].out_features == 32
    assert linear_layers[1].out_features == 16
    assert linear_layers[2].out_features == 8
    assert linear_layers[3].out_features == 2
