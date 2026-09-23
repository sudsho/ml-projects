"""Day 3 - the move step, kernels tuned from the particle cloud, and the genealogy.

Days 1 and 2 moved the cloud with AIS's MALA, identity-preconditioned at a step
of 0.5 / lambda_max of pi_beta's precision. At beta = 1 on the AR(1) that step is
1/227 of the long-axis variance, so a move is a small jitter along the one
direction the cloud most needs to spread in. Two kernels that read their shape
off the resampled cloud instead: random-walk Metropolis with proposal covariance
2.38^2 / d times the cloud's, and MALA preconditioned by the cloud's covariance
at h = 1.65^2 / d^(1/3). Same target and starts, N = 1000, resampling every
step, on day 2's frozen limit schedule at rho = 0.9 and on day 1's uniform
T = 100. Every resampling's parent indices are kept so the final cloud can be
traced back to its starting particles.

1. Tuning the kernel to the cloud is worth more than any number of moves of the
   untuned one. From the CAVI q on the limit schedule (T = 13) one cloud-MALA
   move per temperature is 0.027 +- 0.025 low with a final long-axis variance of
   5.91 of 6.20. Day 1's MALA is 0.60 low at one move and 0.053 +- 0.109 low at
   ten, and needs thirty to reach the target's width. The long-axis
   correlation across one move block says why: 0.957 for day 1's MALA and 0.500
   for cloud MALA, which is at 0.001 by ten moves. From N(0, I) and on T = 100 the
   rows are closer, since there the cloud starts wider or has more temperatures
   to catch up in, and the ordering is the same.

2. Random-walk Metropolis at 27% acceptance beats day 1's MALA at 95%. At three
   moves from the CAVI q it is 0.24 low with variance 4.76 against 0.36 low and
   2.08. The random walk takes big steps in the right shape and is refused
   three times in four. The MALA takes tiny steps in the wrong shape and is
   almost never refused. An acceptance rate near 1 was the symptom, and nothing
   on day 1 or 2 printed it.

3. The genealogy is not a statement about the weights alone. Resampling on
   the same schedule, the final cloud's distinct starting ancestors vary with the
   kernel: from N(0, I) on the limit schedule, 109 of 1000 for one random-walk
   move and 354 for thirty, 179 and 331 for day 1's MALA, against 361 with exact
   draws. A kernel that lags leaves the cloud narrower than pi_beta, the next
   increment's weights are then more uneven than the schedule was built for, and
   more lineages die. Exact draws set the ceiling, 361 to 451 across the four
   settings, and no run on any kernel coalesced to a single ancestor, T = 100
   included.

   The ceiling needs the closed form. The part that can be read without one is
   whether more moves still raise the count. Cloud MALA's goes 376, 423, 438, 436
   from the CAVI q on the limit schedule and has stopped. Day 1's MALA goes 315,
   336, 365, 395 and has not, although its thirty-move variance, 6.19, already
   looks right. As a diagnostic the count agrees with the log Z column about which
   rows have not mixed, and it costs nothing to keep.

4. Two cloud-MALA rows at thirty moves sit 2.1 and 2.6 se below zero, -0.0125
   and -0.0122, when the same kernel at ten moves is on zero in all four settings.
   Ten runs per row give a poor estimate of the se, and 52 rows at 2 se should
   produce two or three such misses anyway. I have not rerun them.

Five predictions written before the run. Three right, two wrong.

- Right: cloud MALA at least 2x less biased than day 1's MALA at one move from
  the CAVI q on the limit schedule. 22x, 0.027 against 0.598.
- Right: cloud MALA inside 2 se of exact draws by ten moves in all four settings.
  The widest gap is 1.4 se, from N(0, I) on the limit schedule.
- Right: no run coalesces to one ancestor, uniform T = 100 included. 0 of 520.
- Wrong: the random-walk kernel needs at least 5x the moves of day 1's MALA for
  the same bias. It needs fewer: 0.068 +- 0.016 low at ten against 0.053 +- 0.109,
  and a long-axis variance of 5.80 against 3.98.
- Wrong: distinct starting ancestors within 20% between one and thirty moves for
  every kernel, since resampling decides the genealogy. 3.3x for the random walk
  from N(0, I).

NumPy only. Fixed seeds. About 5 minutes.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "annealed-importance-sampling"))
sys.path.insert(0, HERE)
from day1_ais_gaussians import GeometricPath, ar1  # noqa: E402
import day1_tempered_smc as day1  # noqa: E402
import day2_adaptive_tempering as day2  # noqa: E402

RNG = np.random.default_rng(33)
day1.RNG = RNG
day2.RNG = RNG


# ----------------------------------------------------------------------------
# Kernels tuned from the cloud
# ----------------------------------------------------------------------------


def log_target(path: GeometricPath, beta: float, x: np.ndarray) -> np.ndarray:
    m, _, p, _ = path.at(beta)
    return -0.5 * np.einsum("ni,ij,nj->n", x - m, p, x - m)


def cloud_rwm(path: GeometricPath, x: np.ndarray, beta: float, steps: int):
    """Random-walk Metropolis with proposal covariance 2.38^2 / d times the cloud's covariance.

    2.38^2 / d is the optimal scale when the proposal shape is the target's.
    The shape here is read off the resampled cloud, so if the cloud is narrower
    than pi_beta the proposal is too, and the kernel explores at the cloud's
    width rather than the target's.
    """
    d = x.shape[1]
    chol = np.linalg.cholesky(np.cov(x.T) * 2.38 ** 2 / d + 1e-10 * np.eye(d))
    lp = log_target(path, beta, x)
    accepted = 0
    for _ in range(steps):
        y = x + RNG.standard_normal(x.shape) @ chol.T
        lp_y = log_target(path, beta, y)
        accept = np.log(RNG.random(len(x))) < lp_y - lp
        x[accept], lp[accept] = y[accept], lp_y[accept]
        accepted += int(accept.sum())
    return x, accepted / (steps * len(x))


def cloud_mala(path: GeometricPath, x: np.ndarray, beta: float, steps: int, h: float = None):
    """MALA preconditioned by the cloud's covariance M, step h = 1.65^2 / d^(1/3).

    With M equal to the target covariance this is MALA on a standard normal,
    where 1.65^2 / d^(1/3) is the optimal step. Day 1's kernel is the same move
    with M = I and h = 0.5 / lambda_max, which at beta = 1 is 1/227 of the
    AR(1)'s long-axis variance.
    """
    d = x.shape[1]
    h = 1.65 ** 2 / d ** (1 / 3) if h is None else h
    m, _, p, _ = path.at(beta)
    cov = np.cov(x.T) + 1e-10 * np.eye(d)
    chol = np.linalg.cholesky(cov)
    inv_chol = np.linalg.inv(chol)
    accepted = 0
    for _ in range(steps):
        drift_x = x - 0.5 * h * (x - m) @ p @ cov
        y = drift_x + np.sqrt(h) * RNG.standard_normal(x.shape) @ chol.T
        drift_y = y - 0.5 * h * (y - m) @ p @ cov
        log_fwd = -np.sum(((y - drift_x) @ inv_chol.T) ** 2, axis=1) / (2 * h)
        log_bwd = -np.sum(((x - drift_y) @ inv_chol.T) ** 2, axis=1) / (2 * h)
        accept = (np.log(RNG.random(len(x)))
                  < log_target(path, beta, y) - log_target(path, beta, x) + log_bwd - log_fwd)
        x[accept] = y[accept]
        accepted += int(accept.sum())
    return x, accepted / (steps * len(x))


def path_mala(path: GeometricPath, x: np.ndarray, beta: float, steps: int):
    x, accepted = day1.mala(path, x, beta, steps)
    return x, accepted / (steps * len(x))


KERNELS = {"path mala": path_mala, "cloud rwm": cloud_rwm, "cloud mala": cloud_mala}


# ----------------------------------------------------------------------------
# The sampler, keeping the genealogy
# ----------------------------------------------------------------------------


def smc(path: GeometricPath, betas: np.ndarray, n: int, kernel, steps: int, axis: np.ndarray):
    """Reweight, resample, move on a fixed schedule, keeping every resampling's parent indices.

    Returns log Z_hat, the final cloud, the parent arrays, the mean correlation
    of the long-axis coordinate across each move block, the mean share of
    distinct positions after it, and the mean acceptance rate.
    """
    x = day2.draw(path, 0.0, n)
    parents, corr, distinct, acc = [], [], [], []
    log_z = 0.0
    for t in range(1, len(betas)):
        inc = (betas[t] - betas[t - 1]) * path.g(x)
        log_z += day2.log_sum_exp(inc) - np.log(n)
        w = np.exp(inc - inc.max())
        idx = day1.systematic(w / w.sum())
        x = x[idx]
        parents.append(idx)
        if t == len(betas) - 1:
            break
        before = x @ axis
        if kernel is None:
            x, a = day2.draw(path, betas[t], n), 1.0
        else:
            x, a = KERNELS[kernel](path, x, betas[t], steps)
        corr.append(np.corrcoef(before, x @ axis)[0, 1])
        distinct.append(len(np.unique(x, axis=0)) / n)
        acc.append(a)
    return log_z, x, parents, np.mean(corr), np.mean(distinct), np.mean(acc)


def genealogy(parents, n: int):
    """(distinct starting ancestors, generations back to one common ancestor or None).

    parents[t][i] is the index before resampling t of the particle that became
    i after it, so following them back from the final cloud gives each final
    particle's ancestor at every generation.
    """
    lineage = np.arange(n)
    depth = None
    for back, idx in enumerate(reversed(parents), start=1):
        lineage = idx[lineage]
        if depth is None and len(np.unique(lineage)) == 1:
            depth = back
    return len(np.unique(lineage)), depth


# ----------------------------------------------------------------------------


def main():
    started = time.time()
    d, n, reps = 8, 1000, 10
    cov = ar1(d, 0.9)
    mean = np.linspace(-1.0, 1.0, d)
    prec = np.linalg.inv(cov)
    starts = {
        "cavi q": GeometricPath(mean, np.diag(1.0 / np.diag(prec)), mean, cov),
        "N(0, I)": GeometricPath(np.zeros(d), np.eye(d), mean, cov),
    }
    axis = np.linalg.eigh(cov)[1][:, -1]
    print(f"target: AR(1) r = 0.9, d = {d}   log Z = {starts['cavi q'].log_z():.6f}"
          f"   long-axis variance {axis @ cov @ axis:.3f}   N = {n}, {reps} runs per row")

    for name, path in starts.items():
        for label, betas in (("limit rho = 0.9", day2.limit_schedule(path, 0.9)),
                             ("uniform", np.linspace(0.0, 1.0, 101))):
            print(f"\n== {name}, {label} schedule, T = {len(betas) - 1} ==")
            print("  kernel      steps   log Z_hat - log Z      sd    long-var   corr   distinct   accept"
                  "   ancestors at 0   coalesced (median depth)")
            for kernel, steps in [(None, 0)] + [(k, s) for k in KERNELS for s in (1, 3, 10, 30)]:
                rows = [smc(path, betas, n, kernel, steps, axis) for _ in range(reps)]
                err = np.array([r[0] for r in rows]) - path.log_z()
                var_long = np.mean([np.var(r[1] @ axis) for r in rows])
                gen = [genealogy(r[2], n) for r in rows]
                depths = [g[1] for g in gen if g[1] is not None]
                med = f"{np.median(depths):5.1f}" if depths else "    -"
                print(f"  {kernel or 'exact':10s}  {steps:4d}   {err.mean():+.4f} +- {err.std(ddof=1) / np.sqrt(reps):.4f}"
                      f"   {err.std(ddof=1):.4f}   {var_long:6.3f}   {np.mean([r[3] for r in rows]):5.3f}"
                      f"   {np.mean([r[4] for r in rows]):6.3f}   {np.mean([r[5] for r in rows]):5.3f}"
                      f"   {np.mean([g[0] for g in gen]):10.1f}      {len(depths):2d}/{reps} ({med})")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
