"""
Day 1: explore the CartPole environment from gymnasium, sanity-check the
observation/action spaces, and run a random-action baseline so we know
what the "no learning" floor looks like before any DQN training.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np


ENV_NAME = "CartPole-v1"
RANDOM_SEED = 7
N_EPISODES = 200
MAX_STEPS_PER_EPISODE = 500  # CartPole-v1 truncates at 500 anyway


@dataclass
class EpisodeRollout:
    total_reward: float
    length: int
    terminated: bool
    truncated: bool


def describe_env(env: gym.Env) -> None:
    print(f"env id            : {ENV_NAME}")
    print(f"observation space : {env.observation_space}")
    print(f"action space      : {env.action_space}")
    low = env.observation_space.low
    high = env.observation_space.high
    print(f"obs low           : {np.array2string(low, precision=3)}")
    print(f"obs high          : {np.array2string(high, precision=3)}")
    # CartPole has 4 obs dims: cart pos, cart vel, pole angle, pole ang vel
    # and 2 discrete actions: 0 = push left, 1 = push right.


def run_random_episode(env: gym.Env, rng: np.random.Generator) -> EpisodeRollout:
    obs, _info = env.reset(seed=int(rng.integers(0, 1 << 31)))
    total = 0.0
    steps = 0
    terminated = False
    truncated = False
    for _ in range(MAX_STEPS_PER_EPISODE):
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, terminated, truncated, _info = env.step(action)
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return EpisodeRollout(total, steps, terminated, truncated)


def summarize(rollouts: List[EpisodeRollout]) -> None:
    rewards = [r.total_reward for r in rollouts]
    lengths = [r.length for r in rollouts]
    print(f"episodes          : {len(rollouts)}")
    print(f"mean reward       : {statistics.mean(rewards):.2f}")
    print(f"stdev reward      : {statistics.stdev(rewards):.2f}")
    print(f"min / max reward  : {min(rewards):.0f} / {max(rewards):.0f}")
    print(f"mean ep length    : {statistics.mean(lengths):.2f}")
    # CartPole-v1 solved threshold is avg >= 475 over 100 episodes,
    # so random play landing around 20 just confirms how far we have to go.


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    env = gym.make(ENV_NAME)
    describe_env(env)

    rollouts: List[EpisodeRollout] = []
    for ep in range(N_EPISODES):
        rollouts.append(run_random_episode(env, rng))
        if (ep + 1) % 50 == 0:
            avg = statistics.mean(r.total_reward for r in rollouts[-50:])
            print(f"  ep {ep + 1:4d} | trailing-50 avg reward = {avg:6.2f}")
    env.close()

    print("\nrandom-action baseline")
    print("-" * 40)
    summarize(rollouts)


if __name__ == "__main__":
    main()
