# Reinforcement Learning with DQN on CartPole

A from-scratch DQN implementation on `CartPole-v1` (gymnasium) followed by a
head-to-head with the Double-DQN variant. The goal here is to build the
classic pieces in the open and watch how the small target-formulation change
affects learning stability.

## Layout

| File | What it does |
|------|--------------|
| `day1_env_explore.py` | Sanity-check the env: obs/action spaces, random-action baseline returns |
| `q_network.py` | `QNetwork` (small MLP) and `ReplayBuffer` (numpy-backed, uniform sampler) |
| `test_replay_buffer.py` | Unit tests covering capacity, wrap-around, sample shape |
| `day3_train.py` | Vanilla DQN training loop: epsilon-greedy, target network, Huber TD loss |
| `day4_double_dqn.py` | Double-DQN variant and `DQN vs DDQN` comparison plot |

## How to run

```bash
python q_network.py            # smoke test
python -m pytest test_replay_buffer.py
python day3_train.py           # train vanilla DQN, write logs/dqn_returns.csv
python day4_double_dqn.py      # train both, write csvs + comparison png
```

## Method notes

### Vanilla DQN

The td target is

```
y = r + gamma * max_a' Q_target(s', a')
```

Online net selects actions, target net is hard-synced every `TARGET_SYNC=500`
steps. Loss is Huber (smooth L1) so early wild Q estimates do not blow up the
gradient. Adam at `1e-3` with grad-norm clipping at 1.0.

Epsilon decays linearly from 1.0 to 0.05 over 10k env steps, warm-up of 1k
steps before any learning updates.

### Double DQN

Same loop, only the target changes:

```
a* = argmax_a' Q_online(s', a')
y  = r + gamma * Q_target(s', a*)
```

Online picks, target evaluates. This removes the upward bias from doing both
operations on the same noisy network. On CartPole the absolute reward ceiling
is the same (500), so the effect shows up as steadier late-training returns
rather than a higher cap.

## Results

Single-seed run (seed=0, 400 episodes each):

| Variant | Final 20-ep mean return |
|---------|-------------------------|
| DQN | ~475 |
| Double DQN | ~492 |

Both solve the env (return >= 475 averaged over the trailing window). Double
DQN gets there with noticeably less variance in the last quarter of training -
see `logs/dqn_vs_double_dqn.png`.

## Lessons

- Huber loss matters early - L2 with the same setup repeatedly diverged.
- The target network sync interval is the single most sensitive hyper. Going
  to `TARGET_SYNC=100` made vanilla DQN oscillate badly; `500` worked across
  seeds.
- For a 4-dim observation a `(128, 128)` MLP is already overkill; the bias
  toward bigger networks does not help on this env.

## Tech Stack

PyTorch, gymnasium, numpy, matplotlib.
