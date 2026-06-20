"""Quick sanity tests for the day-3 NT-Xent loss.

Run with `pytest test_ntxent.py` or just `python test_ntxent.py`.
"""

import torch

from day3_ntxent_loss import NTXentLoss, ntxent_reference


def _stacked_batch(n=5, dim=64, jitter=0.05, seed=0):
    torch.manual_seed(seed)
    z1 = torch.randn(n, dim)
    z2 = z1 + jitter * torch.randn(n, dim)
    return torch.cat([z1, z2], dim=0)


def test_matches_reference():
    z = _stacked_batch()
    fast = NTXentLoss(temperature=0.5)(z)
    ref = ntxent_reference(z, temperature=0.5)
    assert torch.allclose(fast, ref, atol=1e-5)


def test_correlated_views_beat_random():
    z = _stacked_batch(jitter=0.02)
    aligned = NTXentLoss(temperature=0.5)(z)
    torch.manual_seed(1)
    z_rand = torch.randn(z.shape[0], z.shape[1])
    random = NTXentLoss(temperature=0.5)(z_rand)
    assert random > aligned


def test_odd_batch_rejected():
    bad = torch.randn(7, 16)
    try:
        NTXentLoss()(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on an odd number of rows")


if __name__ == "__main__":
    test_matches_reference()
    test_correlated_views_beat_random()
    test_odd_batch_rejected()
    print("all NT-Xent tests passed")
