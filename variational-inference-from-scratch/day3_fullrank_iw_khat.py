"""Day 3 - full-rank against mean-field, the importance-weighted bound, and k-hat.

Day 2 ended on the ELBO being unable to report its own sampling noise. Today is
the part of the plan that is supposed to answer that. Importance weights
`w = p~ / q` are computed from q's samples too, but their tail is a statement
about the directions where q is too narrow, and Pareto-smoothed importance
sampling reads a number off that tail. Every q and every target in this file is
Gaussian, so the tail shape is not something to estimate: it is `k = 1 - m`,
with `m` the smallest eigenvalue of `Lambda_q^-1/2 Lambda_p Lambda_q^-1/2`, and
k-hat can be scored against it.

What is measured.

1. Full-rank against mean-field on AR(1) r = 0.9, d = 8, both fitted by the day
   2 pathwise gradient through the same code. Mean-field lands on its
   closed-form optimum (KL 2.6156 against 2.6103) with tail shape k = 0.981, and
   importance sampling from it misses log Z by -0.99 at S = 4000. Full-rank, 44
   parameters against 16, gets KL down to 1.2e-02 and log Z to rmse 0.0025, and
   still has a positive tail, k = 0.092. Optimisation noise left one direction
   slightly too narrow, and the tail registers it with a KL over 200x smaller.
   k-hat reads 0.891 and 0.135.

2. k-hat against the exact shape, on the d = 2 mean-field optimum where `k = r`.
   Right at r = 0.3 (0.30 to 0.33) and increasingly low above it - 0.80 against
   0.90 at r = 0.9 - and more samples barely move it: 0.786, 0.782, 0.802 at
   S = 1000, 4000, 16000.

3. Two q's the ELBO cannot tell apart. `q = N(mu, s^2 Sigma)` at s = 0.7 and
   s = 1.3341 have identical KL at every dimension, one with a heavy tail
   (k = 0.51) and one with bounded weights. At d = 2 k-hat separates them the
   way the tails say, 0.481 against -0.906, the second next to its endpoint
   value -2/d. That is the sentence in the plan and at d = 2 it holds.

   At d = 32 it does not, and this is the thing to keep. The bounded q reads
   0.531 - over 0.5, where the weights would have infinite variance if the
   number were the tail, for weights that have every moment. It is reading the
   sample: 0.542, 0.466, 0.412 as S goes 4000, 16000, 64000 with nothing else
   changed. The heavy one reads 0.97 against an exact 0.51 and moves *up* with
   S. I had k-hat down as a diagnostic of the tail rather than of the fit. It is
   a diagnostic of the tail as it looks at this sample size, and in 32
   dimensions that is not the tail.

   The direction of the miss is the part I can account for. A Gaussian weight
   tail is Pareto with a power of `log t` in front, set by how many directions
   share the narrowest coefficient and how many go the other way. 32 shared
   directions put `(log t)^15` there and push k-hat up; the d = 2 mean-field
   case has one heavy direction and one light one and pulls it down; the d = 2
   scale family has two shared, a power of zero, and is right. That matches the
   sign of every heavy-tailed row here. I have not checked it in magnitude.

4. The importance-weighted bound. At K = 1 the gap is KL(q || p) exactly, which
   checks the estimators before anything is read from them: plain Monte Carlo
   agrees at every r, and the control-variate version I wrote to cut its noise
   gives 1.158 against 0.830 at r = 0.9, because `x - 1` has no variance once
   `E[w^2]` diverges. So the closed form picks the estimator. The gap should
   close as `K^-min(1, 1/k - 1)`, and over K = 8..1024 the slope is -0.973
   against -1 at r = 0.3 (where `Var(w)/2K` puts the K = 1024 gap at 6.714e-05
   and it measures 6.658e-05), -0.666 against -0.667 at r = 0.6, and -0.189
   against -0.111 at r = 0.9. The last closes 1.7x faster than the tail allows
   over this range, which is the same log factor seen from the bound's side.

   At r = 0.9 a thousand samples take the gap from 0.83 to 0.19 nats. A
   mean-field fit that has converged by its own objective is still 0.19 nats
   from log Z with 1024 importance samples helping it, and neither number shows
   up in the ELBO trace.

NumPy only. Fixed seeds. About 30 seconds, most of it the d = 32 sweep.
"""

from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Gaussian algebra. Every q in this file is Gaussian and so is every target, so
# the tail of the importance weights can be written down before sampling them.
# ----------------------------------------------------------------------------


def ar1(d: int, r: float) -> np.ndarray:
    """Unit-variance covariance with corr(i, j) = r ** |i - j|. Same as days 1-2."""
    idx = np.arange(d)
    return r ** np.abs(idx[:, None] - idx[None, :])


def log_z(cov: np.ndarray) -> float:
    """log of the normaliser of exp(-0.5 (z-mu)' Sigma^-1 (z-mu))."""
    d = cov.shape[0]
    _, logdet = np.linalg.slogdet(cov)
    return 0.5 * (d * np.log(2.0 * np.pi) + logdet)


def kl_gaussians(q_mean, q_cov, mean, cov) -> float:
    """Exact KL(q || p) for two full-covariance Gaussians."""
    prec = np.linalg.inv(cov)
    d = len(mean)
    delta = q_mean - mean
    _, logdet_p = np.linalg.slogdet(cov)
    _, logdet_q = np.linalg.slogdet(q_cov)
    return 0.5 * (
        float(np.trace(prec @ q_cov)) + float(delta @ prec @ delta) - d + logdet_p - logdet_q
    )


def tail_shape(q_cov: np.ndarray, cov: np.ndarray) -> float:
    """The Pareto shape of the weight tail, from the two covariances alone.

    `E_q[w^a]` is a Gaussian integral with precision `a Lambda_p - (a-1) Lambda_q`,
    finite for `a > 1` exactly when that matrix is positive definite, and the
    means only shift a linear term so they do not enter. With `m` the smallest
    eigenvalue of `Lambda_q^-1/2 Lambda_p Lambda_q^-1/2`, moments exist up to
    `a < 1 / (1 - m)`, and a tail with moments up to `1/k` has Pareto shape `k`.
    So `k = 1 - m` when `m < 1`. When `m >= 1` q is at least as wide as p in every
    direction and the weights are bounded; the return value is then negative and
    is not the shape - the caller has to handle that case separately.
    """
    evals, evecs = np.linalg.eigh(np.linalg.inv(q_cov))
    lam_q_inv_sqrt = evecs @ np.diag(evals**-0.5) @ evecs.T
    m = np.linalg.eigvalsh(lam_q_inv_sqrt @ np.linalg.inv(cov) @ lam_q_inv_sqrt).min()
    return float(1.0 - m)


def weight_second_moment(q_cov: np.ndarray, cov: np.ndarray) -> float:
    """E_q[(p/q)^2] for normalised p and q with the same mean, inf if it diverges.

    `int p^2 / q = |Q|^1/2 |Sigma|^-1 |2 Lambda - Q^-1|^-1/2`. Minus one it is the
    variance of the normalised weight, which is what the importance-weighted
    bound's gap is proportional to when it is finite.
    """
    inner = 2.0 * np.linalg.inv(cov) - np.linalg.inv(q_cov)
    if np.linalg.eigvalsh(inner).min() <= 0:
        return float("inf")
    _, logdet_q = np.linalg.slogdet(q_cov)
    _, logdet_p = np.linalg.slogdet(cov)
    _, logdet_inner = np.linalg.slogdet(inner)
    return float(np.exp(0.5 * logdet_q - logdet_p - 0.5 * logdet_inner))


def log_weights(z, q_mean, q_cov, mean, cov) -> np.ndarray:
    """log p~(z) - log q(z), with p~ unnormalised so that E_q[w] = Z."""
    prec_p = np.linalg.inv(cov)
    prec_q = np.linalg.inv(q_cov)
    dp = z - mean
    dq = z - q_mean
    _, logdet_q = np.linalg.slogdet(q_cov)
    d = len(mean)
    log_p = -0.5 * np.einsum("ij,jk,ik->i", dp, prec_p, dp)
    log_q = -0.5 * np.einsum("ij,jk,ik->i", dq, prec_q, dq) - 0.5 * (
        d * np.log(2.0 * np.pi) + logdet_q
    )
    return log_p - log_q


def sample(rng, n, q_mean, q_cov) -> np.ndarray:
    chol = np.linalg.cholesky(q_cov)
    return q_mean + rng.standard_normal((n, len(q_mean))) @ chol.T


# ----------------------------------------------------------------------------
# Pareto-smoothed importance sampling, from scratch
# ----------------------------------------------------------------------------


def gpd_fit(exceedances: np.ndarray) -> Tuple[float, float]:
    """Zhang and Stephens (2009) fit of a generalised Pareto, as PSIS uses it.

    A grid of candidate `b = -k / sigma` values placed from the data, the profile
    likelihood of each, and the posterior mean of `b` under those weights, then
    `k` read back from `b`. The last line is the weakly informative prior PSIS
    puts on `k`, which pulls towards 0.5 with ten pseudo-observations.
    `exceedances` must be positive and sorted ascending.
    """
    x = exceedances
    n = len(x)
    m_est = 30 + int(np.sqrt(n))
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b /= 3.0 * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    profile = n * (np.log(-b / k) - k - 1.0)
    with np.errstate(over="ignore"):
        weights = 1.0 / np.exp(profile[None, :] - profile[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    weights, b = weights[keep], b[keep]
    weights /= weights.sum()
    b_post = float(np.sum(b * weights))
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    k_post = (n * k_post + 10 * 0.5) / (n + 10)
    return k_post, sigma


def psis(log_w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Smoothed log weights and k-hat.

    The largest `min(S/5, 3 sqrt(S))` weights are fitted with a generalised
    Pareto above the next one down, replaced by that fit's quantiles at the
    midpoints, and truncated at the largest raw weight. k-hat is the fitted
    shape, and it is computed from the tail alone - the bulk of the sample,
    which is where q and p agree, does not enter.
    """
    s = len(log_w)
    tail_len = int(np.ceil(min(0.2 * s, 3.0 * np.sqrt(s))))
    shift = log_w.max()
    lw = log_w - shift
    order = np.argsort(lw)
    tail_idx = order[-tail_len:]
    cutoff = np.exp(lw[order[-tail_len - 1]])
    exceed = np.exp(lw[tail_idx]) - cutoff
    k_hat, sigma = gpd_fit(exceed)
    p = (np.arange(1, tail_len + 1) - 0.5) / tail_len
    if abs(k_hat) > 1e-12:
        quantiles = cutoff + sigma / k_hat * ((1.0 - p) ** (-k_hat) - 1.0)
    else:
        quantiles = cutoff - sigma * np.log1p(-p)
    smoothed = lw.copy()
    smoothed[tail_idx] = np.log(np.minimum(quantiles, 1.0))
    return smoothed + shift, k_hat


def log_mean_exp(a: np.ndarray, axis=None) -> np.ndarray:
    top = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.mean(np.exp(a - top), axis=axis, keepdims=True)) + top
    return np.squeeze(out, axis=axis) if axis is not None else float(out.squeeze())


# ----------------------------------------------------------------------------
# Full-rank and mean-field q, fitted by the day 2 reparameterised gradient
# ----------------------------------------------------------------------------


def fit_gaussian_q(mean, prec, rng, mean_field, schedule, n_samples=32):
    """Stochastic gradient ascent on the ELBO over q = N(m, L L').

    L is lower triangular with its diagonal held as a log. The pathwise gradient
    is `E[grad_z log p~ eps']` in the lower triangle plus the exact entropy term
    `diag(1 / L_ii)`, as on day 2 with the off-diagonal added. `mean_field`
    zeroes the off-diagonal gradient, so both fits run through the same code and
    see the same noise. `schedule` is a list of `(steps, lr)`.
    """
    d = len(mean)
    m = mean + 2.0
    raw = np.zeros((d, d))
    for steps, lr in schedule:
        for _ in range(steps):
            chol = np.tril(raw, -1) + np.diag(np.exp(np.diag(raw)))
            eps = rng.standard_normal((n_samples, d))
            z = m + eps @ chol.T
            g_z = -(z - mean) @ prec
            g_chol = (g_z[:, :, None] * eps[:, None, :]).mean(axis=0)
            g_raw = np.zeros((d, d)) if mean_field else np.tril(g_chol, -1)
            g_raw[np.diag_indices(d)] = np.diag(g_chol) * np.diag(chol) + 1.0
            m = m + lr * g_z.mean(axis=0)
            raw = raw + lr * g_raw
    chol = np.tril(raw, -1) + np.diag(np.exp(np.diag(raw)))
    return m, chol @ chol.T


# ----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=4, suppress=True)
    results = {}

    # 1. full-rank against mean-field on a correlated target
    print("== full-rank against mean-field, AR(1) r = 0.9, d = 8 ==")
    d, r = 8, 0.9
    cov = ar1(d, r)
    prec = np.linalg.inv(cov)
    mean = np.linspace(-1.0, 1.0, d)
    schedule = [(5000, 0.05), (3000, 0.005)]
    fits = {}
    for label, mf in (("mean-field", True), ("full-rank ", False)):
        rng = np.random.default_rng(31)
        q_mean, q_cov = fit_gaussian_q(mean, prec, rng, mf, schedule)
        fits[label.strip()] = (q_mean, q_cov)
        n_params = 2 * d if mf else d + d * (d + 1) // 2
        print(
            f"  {label}  params {n_params:3d}   KL(q||p) {kl_gaussians(q_mean, q_cov, mean, cov):.4e}"
            f"   tail shape k {tail_shape(q_cov, cov):+.4f}"
        )
    mf_exact_cov = np.diag(1.0 / np.diag(prec))
    kl_mf_exact = kl_gaussians(mean, mf_exact_cov, mean, cov)
    print(f"  mean-field optimum in closed form: KL {kl_mf_exact:.4e}, k {tail_shape(mf_exact_cov, cov):+.4f}")
    results["kl_mf_exact"] = kl_mf_exact
    results["kl_fr"] = kl_gaussians(*fits["full-rank"], mean, cov)
    results["k_fr"] = tail_shape(fits["full-rank"][1], cov)

    rng = np.random.default_rng(32)
    true_log_z = log_z(cov)
    print("  importance sampling log Z from each fit, S = 4000, 200 seeds")
    for label, (q_mean, q_cov) in fits.items():
        errs, khats = [], []
        for _ in range(200):
            z = sample(rng, 4000, q_mean, q_cov)
            lw = log_weights(z, q_mean, q_cov, mean, cov)
            errs.append(log_mean_exp(lw) - true_log_z)
            khats.append(psis(lw)[1])
        errs = np.array(errs)
        print(
            f"    {label:10s}  bias {errs.mean():+.4f}  rmse {np.sqrt(np.mean(errs**2)):.4f}"
            f"   k-hat {np.mean(khats):.3f} +- {np.std(khats):.3f}"
        )

    # 2. k-hat against the tail shape it estimates, where that shape is known
    print("\n== k-hat against the exact tail shape, d = 2 mean-field optimum (k = r) ==")
    print("   r    exact k   S=1000           S=4000           S=16000")
    khat_table = {}
    for r2 in (0.3, 0.5, 0.7, 0.9):
        cov2 = ar1(2, r2)
        q_cov2 = np.diag(1.0 / np.diag(np.linalg.inv(cov2)))
        exact = tail_shape(q_cov2, cov2)
        row = []
        for s in (1000, 4000, 16000):
            rng = np.random.default_rng(int(1000 * r2) + s)
            ks = [
                psis(log_weights(sample(rng, s, np.zeros(2), q_cov2), np.zeros(2), q_cov2, np.zeros(2), cov2))[1]
                for _ in range(50)
            ]
            row.append((np.mean(ks), np.std(ks)))
        khat_table[r2] = (exact, row)
        cells = "   ".join(f"{mu:.3f} +- {sd:.3f}" for mu, sd in row)
        print(f"  {r2:.1f}   {exact:.3f}    {cells}")
    results["khat_table"] = khat_table

    # 3. two q's the ELBO cannot tell apart
    print("\n== same KL, opposite tails: q = N(mu, s^2 Sigma), s under and over 1 ==")

    def kl_scale(s, dim):
        return 0.5 * dim * (s * s - 1.0 - 2.0 * np.log(s))

    s_under = 0.7
    lo, hi = 1.0 + 1e-9, 3.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if kl_scale(mid, 1) < kl_scale(s_under, 1):
            lo = mid
        else:
            hi = mid
    s_over = 0.5 * (lo + hi)
    print(f"  s = {s_under} and s = {s_over:.4f}; KL per dimension {kl_scale(s_under, 1):.4f} for both")
    print("   d    KL      under: k exact  k-hat            over: bounded  k-hat")
    matched = {}
    for dim in (2, 8, 32):
        cov3 = ar1(dim, 0.5)
        mu3 = np.zeros(dim)
        row = {}
        for label, s in (("under", s_under), ("over", s_over)):
            q_cov3 = s * s * cov3
            rng = np.random.default_rng(dim + (1 if label == "under" else 2))
            ks = [
                psis(log_weights(sample(rng, 4000, mu3, q_cov3), mu3, q_cov3, mu3, cov3))[1]
                for _ in range(40)
            ]
            row[label] = (tail_shape(q_cov3, cov3), np.mean(ks), np.std(ks))
        matched[dim] = row
        print(
            f"  {dim:3d}  {kl_scale(s_under, dim):6.3f}   {row['under'][0]:+.3f}"
            f"        {row['under'][1]:.3f} +- {row['under'][2]:.3f}"
            f"   endpoint -2/d={-2.0 / dim:+.3f}  {row['over'][1]:.3f} +- {row['over'][2]:.3f}"
        )
    # the d = 32 row is the one that needs explaining: if k-hat there is reading
    # the sample rather than the tail, it has to move when only the sample moves
    print("  d = 32 again, same two q's, as the sample grows (20 seeds each)")
    cov3 = ar1(32, 0.5)
    mu3 = np.zeros(32)
    growth = {}
    for s_samples in (4000, 16000, 64000):
        cells = []
        for label, s in (("under", s_under), ("over", s_over)):
            q_cov3 = s * s * cov3
            rng = np.random.default_rng(s_samples + (1 if label == "under" else 2))
            ks = [
                psis(log_weights(sample(rng, s_samples, mu3, q_cov3), mu3, q_cov3, mu3, cov3))[1]
                for _ in range(20)
            ]
            growth[(s_samples, label)] = (np.mean(ks), np.std(ks))
            cells.append(f"{label} {np.mean(ks):.3f} +- {np.std(ks):.3f}")
        print(f"    S={s_samples:6d}   " + "   ".join(cells))
    results["matched"] = matched
    results["growth"] = growth

    # 4. the importance-weighted bound and how fast its gap closes
    print("\n== importance-weighted bound, d = 2 mean-field optimum, gap to log Z ==")
    ks_grid = 2 ** np.arange(0, 11)
    reps, chunk = 16000, 2000
    gaps = {}
    for r4 in (0.3, 0.6, 0.9):
        cov4 = ar1(2, r4)
        q_cov4 = np.diag(1.0 / np.diag(np.linalg.inv(cov4)))
        rng = np.random.default_rng(int(100 * r4))
        sums = np.zeros((2, len(ks_grid)))
        for _ in range(reps // chunk):
            z = sample(rng, chunk * ks_grid[-1], np.zeros(2), q_cov4)
            lw = log_weights(z, np.zeros(2), q_cov4, np.zeros(2), cov4)
            lw = lw.reshape(chunk, ks_grid[-1]) - log_z(cov4)
            # log of the running mean of the normalised weights, per replicate
            running = np.logaddexp.accumulate(lw, axis=1) - np.log(np.arange(1, ks_grid[-1] + 1))
            lx = running[:, ks_grid - 1]
            sums[0] -= lx.sum(axis=0)
            # log x = (x - 1) + [log x - (x - 1)] and x - 1 has mean zero, but it
            # only has a variance if E[w^2] does, so the closed form decides
            # below which of the two estimators is allowed to be read
            sums[1] -= (lx - np.expm1(lx)).sum(axis=0)
        plain, cv = sums / reps
        gaps[r4] = (
            plain,
            cv,
            weight_second_moment(q_cov4, cov4) - 1.0,
            kl_gaussians(np.zeros(2), q_cov4, np.zeros(2), cov4),
        )
    print("  K = 1 is KL(q||p) exactly, so the first row is a check on both estimators")
    for r4 in (0.3, 0.6, 0.9):
        print(
            f"    r={r4:.1f}  exact {gaps[r4][3]:.4e}   plain {gaps[r4][0][0]:.4e}"
            f"   control variate {gaps[r4][1][0]:.4e}"
        )
    chosen = {r4: gaps[r4][1] if np.isfinite(gaps[r4][2]) else gaps[r4][0] for r4 in (0.3, 0.6, 0.9)}
    print("     K    r=0.3 (cv)    r=0.6 (plain)  r=0.9 (plain)")
    for i, k_size in enumerate(ks_grid):
        print(f"  {k_size:4d}   " + "   ".join(f"{chosen[r4][i]:.4e}" for r4 in (0.3, 0.6, 0.9)))
    print("  slope of log gap on log K over K = 8..1024, against the tail's prediction")
    slopes = {}
    fit_idx = ks_grid >= 8
    for r4 in (0.3, 0.6, 0.9):
        exact_k = tail_shape(np.diag(1.0 / np.diag(np.linalg.inv(ar1(2, r4)))), ar1(2, r4))
        predicted = -1.0 if exact_k < 0.5 else -(1.0 / exact_k - 1.0)
        series = chosen[r4]
        slopes[r4] = float(np.polyfit(np.log(ks_grid[fit_idx]), np.log(series[fit_idx]), 1)[0])
        var_w = gaps[r4][2]
        extra = (
            f"   Var(w)/2K at K=1024 {var_w / 2048:.3e} against {series[-1]:.3e}"
            if np.isfinite(var_w)
            else "   Var(w) infinite"
        )
        print(f"    r={r4:.1f}  measured {slopes[r4]:+.3f}   predicted {predicted:+.3f}{extra}")
    results["gaps"] = gaps
    results["slopes"] = slopes
    return results


if __name__ == "__main__":
    main()
