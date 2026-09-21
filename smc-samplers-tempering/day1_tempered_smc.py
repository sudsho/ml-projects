"""Day 1 - tempered SMC on the AIS project's Gaussian targets.

AIS carried each run's weight from q to p on its own. An SMC sampler runs the
same path with the same kernel and, between temperatures, resamples the cloud
by its weights, estimating Z as a product of per-step mean incremental weights
instead of one mean of products. Here AIS is the sampler with resampling turned
off, so the two share every line except the resample. The target is AIS day 1's:
VI day 3's AR(1), r = 0.9, d = 8, `log Z = 1.538949`, long-axis variance 6.203,
from the CAVI q (long-axis variance 0.12) and from N(0, I).

1. With exact moves resampling is worth nothing that can be measured. Each
   SMC increment is an independent mean, so the log of the product is off by
   `sum (r_t - 1) / 2N` to first order, where r_t is the per-step second moment
   AIS day 1 already had, and AIS's log mean of N products is off by
   `(prod r_t - 1) / 2N`. AIS can only be worse, and on this path it is worse by
   the cross terms, 0.0025 against 0.0021 at T = 100, N = 100 from the CAVI q.
   The simulations sit inside 2 se of both and do not separate them. At T = 30
   the CAVI q's last step is past beta* = 1.019 for both samplers, so the one
   infinite increment is shared and SMC has no advantage there either, -0.029
   against -0.029.

2. With MALA moves resampling is most of the answer. From the CAVI q at
   N = 1000 and one step per temperature, AIS's `log mean w` is 0.85, 0.37 and
   0.079 low at T = 10, 100 and 1000, and SMC resampling at every step is 0.64,
   0.19 and 0.050 low. At 1000 gradient steps either way, AIS is 0.113 low with
   ten steps at T = 100 and SMC is 0.032 low with the same split. Since the exact
   rows show the estimator itself buys nothing, the gain is the resample making
   up for a kernel that lags the path.

3. Resampling does move the cloud into the region AIS did not reach, on this
   target. The weighted final cloud's variance along the long axis, 6.20 in p,
   is 1.94 for AIS against 2.75 for SMC at T = 100 with one step, and 3.84
   against 5.38 with ten. It does it from about a third of the starting
   particles: 358 and 420 distinct ancestors of 1000 survive. Killing the
   particles near the centre and letting MALA spread the copies of the far ones
   is a way along a Gaussian's long axis, which diffusion reaches anyway, given
   time. Whether it is a way into the funnel's neck is day 4's question.

4. Resampling only when the ESS drops below N/2 fires one to three times over
   T = 100 and gets about the same bias, 0.135 against 0.190 and 0.084 against
   0.032, none of it resolved. What it costs from the CAVI q is the spread:
   the sd of log Z_hat is 0.51, 0.56, 0.13 and 0.44 across the four settings
   against 0.34, 0.34, 0.10 and 0.11 for resampling every step. From N(0, I)
   it is larger only in the two short one-step runs, 0.48 and 0.21 against 0.37
   and 0.13, and a little smaller in the two long ones. One ESS run at T = 1000 lands well above log Z and puts the row's
   mean at +0.127, which an unbiased estimator of Z is allowed to do.

Five predictions written before the run. One right, one half, three wrong.

- Right: with exact moves both samplers inside 2 se of their closed forms at
  T = 100. The largest miss is 1.8 se.
- Half: ESS-triggered resampling no worse in bias than resampling every step.
  No bias difference resolves; its sd is up to 4x larger from the CAVI q.
- Wrong: SMC with exact moves at least 3x less biased than AIS at T = 100. The
  closed forms are 1.2x apart and the simulation cannot tell them apart.
- Wrong: SMC at least 2x less biased than AIS with one MALA step at T = 100.
  It is 1.97x, a miss by the letter, with a standard error of 0.075 on the SMC
  row that does not resolve it either way.
- Wrong: resampling leaves the long-axis variance within 10% of AIS's. It is 42%
  higher with one step and 40% higher with ten.

NumPy only. Fixed seeds. About 3.5 minutes.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "annealed-importance-sampling"))
from day1_ais_gaussians import GeometricPath, ar1, exact_log_weight, log_mean_exp  # noqa: E402

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------
# Closed forms for exact transitions
# ----------------------------------------------------------------------------


def step_second_moments(path: GeometricPath, betas: np.ndarray) -> np.ndarray:
    """log E[w_t^2] / E[w_t]^2 for each increment w_t = f_t / f_{t-1} under pi_{t-1}.

    The same extrapolated normaliser day 1 of AIS summed into its second moment,
    kept per step instead. inf where `2 beta_t - beta_{t-1}` is past beta*.
    """
    out = np.empty(len(betas) - 1)
    for t in range(1, len(betas)):
        ext = path.at(2.0 * betas[t] - betas[t - 1])
        if ext is None:
            out[t - 1] = np.inf
        else:
            out[t - 1] = ext[3] + path.at(betas[t - 1])[3] - 2.0 * path.at(betas[t])[3]
    return out


def delta_bias(log_r: np.ndarray, n: int, smc: bool) -> float:
    """First-order bias of the log of an N-particle estimate of Z.

    SMC with exact moves multiplies T independent means, each off by
    `(r_t - 1) / 2N` in log. AIS takes one mean of N products, off by
    `(prod r_t - 1) / 2N`. Since prod r >= 1 + sum (r - 1), AIS can only be
    worse, and the gap is the cross terms of the product.
    """
    if not np.all(np.isfinite(log_r)):
        return np.inf
    if smc:
        return float(np.sum(np.expm1(log_r))) / (2 * n)
    return float(np.expm1(np.sum(log_r))) / (2 * n)


# ----------------------------------------------------------------------------
# One sampler: AIS is the SMC sampler that never resamples
# ----------------------------------------------------------------------------


def systematic(weights: np.ndarray) -> np.ndarray:
    """Systematic resampling indices, one uniform for the whole cloud."""
    n = len(weights)
    u = (RNG.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(weights), u), n - 1)


def mala(path: GeometricPath, x: np.ndarray, beta: float, steps: int, scale: float = 0.5):
    """`steps` MALA moves targeting pi_beta, the kernel ais_mala used, as a function."""
    m, _, p, _ = path.at(beta)
    h = scale / np.linalg.eigvalsh(p)[-1]
    accepted = 0
    for _ in range(steps):
        grad = -(x - m) @ p
        y = x + 0.5 * h * grad + np.sqrt(h) * RNG.standard_normal(x.shape)
        grad_y = -(y - m) @ p
        log_target = (-0.5 * np.einsum("ni,ij,nj->n", y - m, p, y - m)
                      + 0.5 * np.einsum("ni,ij,nj->n", x - m, p, x - m))
        log_fwd = -np.sum((y - x - 0.5 * h * grad) ** 2, axis=1) / (2 * h)
        log_bwd = -np.sum((x - y - 0.5 * h * grad_y) ** 2, axis=1) / (2 * h)
        accept = np.log(RNG.random(len(x))) < log_target + log_bwd - log_fwd
        x[accept] = y[accept]
        accepted += int(accept.sum())
    return x, accepted


def tempered(path: GeometricPath, betas: np.ndarray, n: int, move, resample: str):
    """Reweight, maybe resample, move. resample is 'never', 'always' or 'ess'.

    log Z is accumulated as `log sum_i W_i exp(inc_i)` at each step, with W the
    normalised weights carried in. With 'never' the W's are the running AIS
    weights and the telescoped sum is exactly `log mean w`, so AIS is not a
    separate code path. `move` is None for exact draws from pi_beta, else a
    number of MALA steps. Returns log Z_hat, the weighted final cloud, the
    number of resampling events and the count of distinct starting ancestors.
    """
    m0, s0, _, _ = path.at(0.0)
    x = m0 + RNG.standard_normal((n, path.d)) @ np.linalg.cholesky(s0).T
    log_w = np.zeros(n)
    ancestor = np.arange(n)
    log_z = 0.0
    events = 0
    for t in range(1, len(betas)):
        inc = (betas[t] - betas[t - 1]) * path.g(x)
        log_norm = log_w - log_mean_exp(log_w) - np.log(n)
        log_z += log_mean_exp(log_norm + inc) + np.log(n)
        log_w = log_w + inc
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        if resample == "always" or (resample == "ess" and 1.0 / np.sum(w * w) < n / 2):
            idx = systematic(w)
            x, ancestor, log_w = x[idx], ancestor[idx], np.zeros(n)
            events += 1
        if t == len(betas) - 1:
            break
        if move is None:
            m, s, _, _ = path.at(betas[t])
            x = m + RNG.standard_normal((n, path.d)) @ np.linalg.cholesky(s).T
        else:
            x, _ = mala(path, x, betas[t], move)
    w = np.exp(log_w - log_w.max())
    return log_z, x, w / w.sum(), events, len(np.unique(ancestor))


def repeat(path, betas, n, move, resample, reps, axis):
    """Bias and sd of log Z_hat over `reps` independent samplers, plus cloud stats."""
    est, var_long, anc, ev = [], [], [], []
    for _ in range(reps):
        lz, x, w, events, ancestors = tempered(path, betas, n, move, resample)
        est.append(lz)
        proj = x @ axis
        mu = np.sum(w * proj)
        var_long.append(np.sum(w * (proj - mu) ** 2))
        anc.append(ancestors)
        ev.append(events)
    est = np.array(est) - path.log_z()
    return est.mean(), est.std(ddof=1), np.mean(var_long), np.mean(anc), np.mean(ev)


# ----------------------------------------------------------------------------


def main():
    started = time.time()
    d = 8
    cov = ar1(d, 0.9)
    mean = np.linspace(-1.0, 1.0, d)
    prec = np.linalg.inv(cov)
    starts = {
        "cavi q": GeometricPath(mean, np.diag(1.0 / np.diag(prec)), mean, cov),
        "N(0, I)": GeometricPath(np.zeros(d), np.eye(d), mean, cov),
    }
    axis = np.linalg.eigh(cov)[1][:, -1]
    print(f"target: AR(1) r = 0.9, d = {d}   log Z = {starts['cavi q'].log_z():.6f}"
          f"   long-axis variance {axis @ cov @ axis:.3f}")

    print("\n== exact moves: closed-form first-order bias of log Z_N, and simulated ==")
    for name, path in starts.items():
        for T, n, reps in ((30, 100, 400), (100, 100, 400), (100, 1000, 100)):
            betas = np.linspace(0.0, 1.0, T + 1)
            log_r = step_second_moments(path, betas)
            _, _, _, log_second = exact_log_weight(path, betas)
            assert np.isinf(log_second) == np.isinf(log_r.sum())
            if np.isfinite(log_second):
                assert abs(log_second - log_r.sum()) < 1e-9
            rows = []
            for resample in ("always", "never"):
                b, s, _, _, _ = repeat(path, betas, n, None, resample, reps, axis)
                rows.append((b, s))
            print(f"  {name:8s} T={T:4d} N={n:5d}   SMC {rows[0][0]:+.4f} +- {rows[0][1] / np.sqrt(reps):.4f}"
                  f" (delta {-delta_bias(log_r, n, True):+.4f}, sd {rows[0][1]:.4f})"
                  f"   AIS {rows[1][0]:+.4f} +- {rows[1][1] / np.sqrt(reps):.4f}"
                  f" (delta {-delta_bias(log_r, n, False):+.4f}, sd {rows[1][1]:.4f})"
                  f"   worst step log r {log_r.max():.4f}")

    print("\n== MALA moves, N = 1000, same kernel and gradient count for every row ==")
    print("  start     T  steps  resample   log Z_hat - log Z     sd    long-axis var  ancestors  events")
    for name, path in starts.items():
        for T, steps in ((10, 1), (100, 1), (100, 10), (1000, 1)):
            betas = np.linspace(0.0, 1.0, T + 1)
            reps = 20 if T * steps <= 100 else 10
            for resample in ("never", "ess", "always"):
                b, s, v, a, e = repeat(path, betas, 1000, steps, resample, reps, axis)
                print(f"  {name:8s} {T:4d}  {steps:3d}   {resample:6s}   {b:+.4f} +- {s / np.sqrt(reps):.4f}"
                      f"   {s:6.4f}   {v:8.3f}     {a:7.1f}   {e:6.1f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
