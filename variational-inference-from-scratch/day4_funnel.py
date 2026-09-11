"""Day 4 - VI on Neal's funnel, next to the MCMC project's day 4.

MCMC day 4 ran HMC on this target and measured the diagnostics instead of the
sampler. The marginal variance came back within 8% on chains that never reached
the neck, R-hat certified them, and the divergence count was the one number that
tracked the difficulty - 0, 21, 113, 4204 as sigma_v went from 1 to 4 - because a
divergence is produced in the region the chain fails to visit. Day 3 put k-hat
forward as the VI counterpart, computed from the weight tail, which is where q is
too narrow, and then found that at d = 32 it reads the sample and not the tail.
Today uses MCMC's target at MCMC's d = 3, so the two projects' failure reports
can sit in one table.

What is measured.

1. The reverse-KL optimum is closed form. A Gaussian q against the funnel has an
   exact ELBO, because `E[x^2 exp(-v)]` under a joint Gaussian is an exponential
   tilt, and setting its gradient to zero gives every mean 0,
   `1/s_v^2 = 1/sigma_v^2 + k/2` and `s_x^2 = exp(-s_v^2 / 2)`. At sigma_v = 3
   that is Var(v) 0.900 against 9.00 and Var(x_i) 0.638 against 90.02. The gap
   reduces to `KL = 0.5 log(1 + k sigma_v^2 / 2)`, 1.151293, which the
   closed-form ELBO matches to six places and 1e6 draws of q put at 1.150956.
   The variance ratio is `exp(-2 KL)` exactly, so on this family the ELBO gap and
   the variance underestimate are one number written two ways.

   Black-box fits from N(0, I) land on it, exact ELBO -1.15135 mean-field and
   -1.15147 full-rank against -1.15129. Full-rank buys nothing, max |corr| 0.014,
   and that is not the optimiser. `E[x | v] = 0` at every v, so
   `Cov_p(x_i, v) = 0` exactly: the funnel's dependence is not a correlation and
   a Gaussian has nothing to fit to. Forcing corr(x_0, v) into the optimum with
   the tilt cancelled costs 0.00503 nats at 0.1 and 0.04716 at 0.3, which is
   `-0.5 log(1 - rho^2)` to five places - the entropy the correlation removes,
   with nothing bought back in `E_q log p`.

2. The trace. Both fits sit at -1.144 over the second half with no drift beyond
   one standard error (+0.016 and +0.021 between quarters). Its level cannot
   show the 1.151-nat gap, since a q fitting a target with log Z = -1.151
   exactly would print the same level.

   Its jitter can, and that is the one thing I would change in day 2's sentence.
   An exact q makes `log p - log q` the constant log Z, so every estimate in the
   trace would be identical. Here the per-draw sd is 1.260, which is 0.079 per
   256-draw estimate against the trace's measured 0.071, and the non-centred fit,
   which is nearly exact, has 0.011. What the jitter does not carry is a scale.
   If log w were Gaussian, `E_q[w] = 1` would pin the gap at
   `Var(log w) / 2 = 0.794`. The gap is 1.151, and the 0.357 between them is the
   weight tail, the part a variance underweights.

3. Where q put no mass, next to where the chain did not go. HMC's centred chains
   missed the neck - min v -7.48 at best and -2.71 at worst - while Var(v) stayed
   near 9, which it cannot do without the mouth. q has neither end. 18k draws
   span v in [-4.16, 4.75]; p has 8.3% of its mass below that and 5.7% above,
   17.1% beyond each of q's 3-sd marks in v, and 20.8% with |x_0| outside q's
   3 sd. The variance ratio is 0.10 against the chain's 0.73.

4. k-hat, three q's at S = 4000 over 40 seeds. The reverse-KL optimum reads
   0.880 +- 0.122 and the moment-matched q, which has the funnel's true
   covariance, reads 0.916 +- 0.138. The ELBO puts them 8096 nats apart, -1.15
   against -8097.58. As importance proposals they rank the other way: plain IS
   misses log Z by -0.330 from the reverse-KL fit and by -0.044 from the
   moment-matched one, 7x less. k-hat cannot tell them apart. The non-centred
   fit reads 0.046 with log Z to 0.0000 +- 0.0002.

   PSIS moves both centred estimates further from the truth, -0.463 against
   -0.330 and -0.144 against -0.044, while cutting their spread. It replaces the
   largest weights with a fitted tail truncated at the largest raw weight, and on
   a target whose missing mass lives in weights larger than any that were drawn,
   that pulls toward the answer q already had.

   The exact shape those readings estimate is 1, for every Gaussian q in the
   centred coordinates. The x-integral inside `E_q[w^a | v]` has precision
   `(1 - a)/S_xx + a exp(-v)`, which goes negative once v is large enough for any
   a > 1, and a Gaussian q puts mass at every v. No q in the family has a finite
   weight moment above the first, so the estimand says nothing about which q is
   better. Which one is heavier depends on which moment is asked about: at a = 2,
   40% of the reverse-KL q sits where the conditional moment is already infinite
   against 4.2% of the moment-matched one, and at a = 1.01 it is 5.7e-06 against
   1.2e-03, the other way round by 200x.

   Against that exact 1 the reverse-KL q reads 0.808, 0.808, 0.840, 0.872 at
   S = 1000 to 64000, climbing slowly, which is day 3's d = 32 finding in three
   dimensions. The same S = 4000 over 20 fresh seeds reads 0.808 against the
   40-seed 0.880 above, and that is the size of the problem: one reading's sd,
   0.12 to 0.13, is as large as the differences k-hat is being asked to rank.

5. sigma_v swept, q at its closed-form optimum, beside MCMC day 4's columns.

     sigma_v   q Var(v)/true   KL      k-hat (40 seeds)   IS log Z   HMC div   HMC Var ratio
       1          0.500        0.347   0.662 +- 0.112      -0.017         0       1.017
       2          0.200        0.805   0.786 +- 0.117      -0.163        21       0.948
       3          0.100        1.151   0.800 +- 0.128      -0.444       113       0.728
       4          0.059        1.417   0.861 +- 0.111      -0.554      4204       0.765

   k-hat is monotone, like the divergence count and unlike the chain's variance
   ratio, and it is computed from the edge of where q went the way the count is
   computed from where the chain failed to go. But it spans 0.66 to 0.86 with a
   one-reading sd of 0.11 to 0.13, where the count spans three orders of
   magnitude. Averaged over 40 seeds it ranks the four problems. A single run,
   which is what anyone actually has, puts sigma_v = 2 and 4 in the right order
   about two times in three if the readings are roughly normal.

   The prediction written into this file before the run was that k-hat would
   track `k_v = 1 - s_v^2 / sigma_v^2`, day 3's Gaussian formula applied along v
   alone: 0.500, 0.800, 0.900, 0.941. It does not. It sits above k_v at
   sigma_v = 1 and below it from 3 on, pulled in from both ends, and it does not
   track the exact k = 1 either. It is the tail as 4000 draws see it, again.

NumPy only. Fixed seeds. About 3 seconds.
"""

import math
import time
from typing import Callable, Tuple

import numpy as np

from day3_fullrank_iw_khat import log_mean_exp, psis

# MCMC day 4, section [6]: HMC at eps = 0.2, L = 20, 12k steps, sigma_v swept.
# Quoted rather than imported so this file does not reach into another project.
MCMC_DIVERGENCES = {1.0: 0, 2.0: 21, 3.0: 113, 4.0: 4204}
MCMC_VAR_RATIO = {1.0: 1.017, 2.0: 0.948, 3.0: 0.728, 4.0: 0.765}


# ----------------------------------------------------------------------------
# The funnel, normalised, so log Z = 0 and every importance estimate is its own
# error
# ----------------------------------------------------------------------------


def funnel_log_p(z: np.ndarray, sigma_v: float) -> np.ndarray:
    """`v ~ N(0, sigma_v^2)`, `x_i | v ~ N(0, exp(v))`, v in the last column.

    The same target as MCMC day 3's `Funnel`, with the constants kept so that
    it integrates to one.
    """
    v, x = z[:, -1], z[:, :-1]
    k = x.shape[1]
    return (
        -0.5 * (v / sigma_v) ** 2
        - 0.5 * np.log(2.0 * np.pi * sigma_v**2)
        - 0.5 * np.sum(x * x, axis=1) * np.exp(-v)
        - 0.5 * k * v
        - 0.5 * k * np.log(2.0 * np.pi)
    )


def funnel_grad(z: np.ndarray, sigma_v: float) -> np.ndarray:
    v, x = z[:, -1], z[:, :-1]
    g = np.empty_like(z)
    e = np.exp(-v)
    g[:, :-1] = -x * e[:, None]
    g[:, -1] = -v / sigma_v**2 + 0.5 * np.sum(x * x, axis=1) * e - 0.5 * x.shape[1]
    return g


def noncentred_log_p(z: np.ndarray, sigma_v: float) -> np.ndarray:
    """The same funnel in `(xt, v)` with `x = xt exp(v/2)`: independent
    Gaussians, and the Jacobian of the map is absorbed so this also integrates
    to one."""
    v, xt = z[:, -1], z[:, :-1]
    k = xt.shape[1]
    return (
        -0.5 * (v / sigma_v) ** 2
        - 0.5 * np.log(2.0 * np.pi * sigma_v**2)
        - 0.5 * np.sum(xt * xt, axis=1)
        - 0.5 * k * np.log(2.0 * np.pi)
    )


def noncentred_grad(z: np.ndarray, sigma_v: float) -> np.ndarray:
    g = -z.copy()
    g[:, -1] = -z[:, -1] / sigma_v**2
    return g


def gaussian_log_q(z: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = len(mean)
    chol = np.linalg.cholesky(cov)
    white = np.linalg.solve(chol, (z - mean).T).T
    return (
        -0.5 * np.sum(white * white, axis=1)
        - np.sum(np.log(np.diag(chol)))
        - 0.5 * d * np.log(2.0 * np.pi)
    )


def sample(rng, n: int, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return mean + rng.standard_normal((n, len(mean))) @ np.linalg.cholesky(cov).T


def normal_cdf(a: float) -> float:
    return 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))


# ----------------------------------------------------------------------------
# Closed forms. A Gaussian q against the funnel still has an exact ELBO, because
# E[x^2 exp(-v)] under a joint Gaussian is an exponential tilt.
# ----------------------------------------------------------------------------


def exact_elbo(mean: np.ndarray, cov: np.ndarray, sigma_v: float) -> float:
    """ELBO of any Gaussian q against the normalised funnel, so `-KL(q || p)`.

    Tilting a joint Gaussian by `exp(-v)` moves the mean of `x_i` by `-S_iv` and
    leaves the covariance alone, so
    `E[x_i^2 exp(-v)] = exp(-m_v + S_vv / 2) ((m_i - S_iv)^2 + S_ii)`.
    """
    d = len(mean)
    k = d - 1
    m_v, s_vv = mean[-1], cov[-1, -1]
    tilt = np.exp(-m_v + 0.5 * s_vv)
    e_log_p = (
        -0.5 * (m_v**2 + s_vv) / sigma_v**2
        - 0.5 * np.log(2.0 * np.pi * sigma_v**2)
        - 0.5 * tilt * sum((mean[i] - cov[i, -1]) ** 2 + cov[i, i] for i in range(k))
        - 0.5 * k * m_v
        - 0.5 * k * np.log(2.0 * np.pi)
    )
    _, logdet = np.linalg.slogdet(2.0 * np.pi * np.e * cov)
    return float(e_log_p + 0.5 * logdet)


def mean_field_optimum(sigma_v: float, d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """The reverse-KL Gaussian optimum on the funnel, in closed form.

    Setting the gradient of `exact_elbo` to zero: every mean is 0,
    `s_x^2 = exp(m_v - s_v^2 / 2)`, and then `1 / s_v^2 = 1 / sigma_v^2 + k / 2`.
    """
    k = d - 1
    s_vv = 1.0 / (1.0 / sigma_v**2 + 0.5 * k)
    cov = np.diag([np.exp(-0.5 * s_vv)] * k + [s_vv])
    return np.zeros(d), cov


def moment_matched(sigma_v: float, d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """The Gaussian with the funnel's true mean and covariance. Forward KL's
    optimum within the family."""
    k = d - 1
    return np.zeros(d), np.diag([np.exp(0.5 * sigma_v**2)] * k + [sigma_v**2])


# ----------------------------------------------------------------------------
# Black-box fit, the day 3 pathwise gradient with the target passed in
# ----------------------------------------------------------------------------


def fit_gaussian_q(
    grad_log_p: Callable[[np.ndarray], np.ndarray],
    log_p: Callable[[np.ndarray], np.ndarray],
    d: int,
    rng,
    mean_field: bool,
    schedule,
    n_samples: int = 32,
    trace_every: int = 50,
):
    """Stochastic gradient ascent on the ELBO over q = N(m, L L'), starting at
    N(0, I). Returns `(mean, cov, trace)`, the trace being the ELBO as a
    practitioner has it - a Monte Carlo estimate from 256 fresh draws of q."""
    m = np.zeros(d)
    raw = np.zeros((d, d))
    trace = []
    step = 0
    for steps, lr in schedule:
        for _ in range(steps):
            chol = np.tril(raw, -1) + np.diag(np.exp(np.diag(raw)))
            eps = rng.standard_normal((n_samples, d))
            z = m + eps @ chol.T
            g_z = grad_log_p(z)
            g_chol = (g_z[:, :, None] * eps[:, None, :]).mean(axis=0)
            g_raw = np.zeros((d, d)) if mean_field else np.tril(g_chol, -1)
            g_raw[np.diag_indices(d)] = np.diag(g_chol) * np.diag(chol) + 1.0
            m = m + lr * g_z.mean(axis=0)
            raw = raw + lr * g_raw
            step += 1
            if step % trace_every == 0:
                chol = np.tril(raw, -1) + np.diag(np.exp(np.diag(raw)))
                zt = m + rng.standard_normal((256, d)) @ chol.T
                cov = chol @ chol.T
                trace.append(float(np.mean(log_p(zt) - gaussian_log_q(zt, m, cov))))
    chol = np.tril(raw, -1) + np.diag(np.exp(np.diag(raw)))
    return m, chol @ chol.T, np.array(trace)


def importance_summary(rng, log_p, mean, cov, n_draws, seeds):
    """k-hat, plain and PSIS log Z over `seeds` independent samples. log Z = 0."""
    khats, plain, smoothed = [], [], []
    for _ in range(seeds):
        z = sample(rng, n_draws, mean, cov)
        lw = log_p(z) - gaussian_log_q(z, mean, cov)
        slw, k_hat = psis(lw)
        khats.append(k_hat)
        plain.append(log_mean_exp(lw))
        smoothed.append(log_mean_exp(slw))
    return np.array(khats), np.array(plain), np.array(smoothed)


# ----------------------------------------------------------------------------


def main():
    t0 = time.time()
    d, sigma_v = 3, 3.0
    results = {}

    print("== [1] the reverse-KL Gaussian on the funnel, closed form and fitted ==")
    mean_opt, cov_opt = mean_field_optimum(sigma_v, d)
    kl_opt = -exact_elbo(mean_opt, cov_opt, sigma_v)
    print(f"  closed form: Var(v) {cov_opt[-1, -1]:.4f} against {sigma_v**2:.2f}"
          f"   Var(x_i) {cov_opt[0, 0]:.4f} against {np.exp(0.5 * sigma_v**2):.2f}")
    print(f"  KL(q || p) {kl_opt:.6f}   0.5 log(1 + k sigma^2 / 2) = {0.5 * np.log(1 + sigma_v**2):.6f}")
    rng = np.random.default_rng(40)
    z = sample(rng, 1_000_000, mean_opt, cov_opt)
    mc = np.mean(gaussian_log_q(z, mean_opt, cov_opt) - funnel_log_p(z, sigma_v))
    print(f"  Monte Carlo KL from 1e6 draws of q: {mc:.6f}")
    results["kl_opt"] = kl_opt

    fits = {}
    schedule = [(4000, 0.01), (4000, 0.001)]
    for label, mf in (("mean-field", True), ("full-rank", False)):
        rng = np.random.default_rng(41)
        m, c, trace = fit_gaussian_q(
            lambda zz: funnel_grad(zz, sigma_v), lambda zz: funnel_log_p(zz, sigma_v),
            d, rng, mf, schedule,
        )
        fits[label] = (m, c, trace)
        sd = np.sqrt(np.diag(c))
        corr = c / np.outer(sd, sd)
        off = np.max(np.abs(corr[np.triu_indices(d, 1)]))
        # a random 3-variable correlation for scale on the off-diagonal
        print(f"  {label:10s}  mean {np.array2string(m, precision=3)}"
              f"  var {np.array2string(np.diag(c), precision=3)}"
              f"  max |corr| {off:.3f}   exact ELBO {exact_elbo(m, c, sigma_v):.5f}")
    print(f"  closed-form optimum ELBO {-kl_opt:.5f}")

    # an explicit test of the zero-correlation claim: put correlation into the
    # optimum and watch the exact ELBO fall
    for c_xv in (0.1, 0.3):
        cov_c = cov_opt.copy()
        cov_c[0, -1] = cov_c[-1, 0] = c_xv * np.sqrt(cov_c[0, 0] * cov_c[-1, -1])
        mean_c = mean_opt.copy()
        mean_c[0] = cov_c[0, -1]
        print(f"    corr(x0, v) = {c_xv}, x0 mean moved to cancel the tilt: "
              f"ELBO {exact_elbo(mean_c, cov_c, sigma_v):.5f}")

    print("\n== [2] the trace, as the fit sees it ==")
    for label in ("mean-field", "full-rank"):
        trace = fits[label][2]
        tail = trace[len(trace) // 2:]
        print(f"  {label:10s}  first {trace[:3].round(3)}  last half mean {tail.mean():.4f}"
              f"  sd {tail.std():.4f}  drift (last quarter - second quarter)"
              f" {trace[3 * len(trace) // 4:].mean() - trace[len(trace) // 4: len(trace) // 2].mean():+.4f}")
    print(f"  log Z = 0, so the gap the trace's level cannot show is {kl_opt:.4f} nats")
    # the jitter is the part of the trace that does depend on q being wrong: an
    # exact q makes log p - log q the constant log Z and every estimate identical
    lw_opt = funnel_log_p(z, sigma_v) - gaussian_log_q(z, mean_opt, cov_opt)
    print(f"  per-draw sd of log p - log q at the optimum {lw_opt.std():.4f},"
          f" so {lw_opt.std() / 16:.4f} per 256-draw estimate")
    print(f"  if log w were Gaussian, E_q[w] = 1 would make the gap Var(log w)/2 = {0.5 * lw_opt.var():.4f}")

    print("\n== [3] where q put no mass, next to where the chain did not go ==")
    rng = np.random.default_rng(42)
    draws = sample(rng, 18_000, mean_opt, cov_opt)
    min_v = draws[:, -1].min()
    print(f"  18k draws from q: min v {min_v:.3f}   max v {draws[:, -1].max():.3f}"
          f"   (MCMC day 4 centred HMC min v -7.48 .. -2.71)")
    print(f"  p mass below q's min v: {normal_cdf(min_v / sigma_v):.4f}"
          f"   above its max v: {1 - normal_cdf(draws[:, -1].max() / sigma_v):.4f}")
    s_v = np.sqrt(cov_opt[-1, -1])
    s_x = np.sqrt(cov_opt[0, 0])
    print(f"  p mass outside q's +-3 sd in v: {2 * normal_cdf(-3 * s_v / sigma_v):.4f}  (q's own: 0.0027)")
    v_grid = np.linspace(-12 * sigma_v, 12 * sigma_v, 200_001)
    dv = v_grid[1] - v_grid[0]
    p_v = np.exp(-0.5 * (v_grid / sigma_v) ** 2) / np.sqrt(2 * np.pi * sigma_v**2)
    tail_x = np.array([2 * normal_cdf(-3 * s_x * math.exp(-0.5 * vv)) for vv in v_grid[::100]])
    mass_x = float(np.sum(p_v[::100] * tail_x) * dv * 100)
    print(f"  p mass with |x_0| outside q's +-3 sd: {mass_x:.4f}")
    neck = normal_cdf(-3 * s_v / sigma_v)
    print(f"  split by v: below -3 s_v (the neck) {neck:.4f}, above +3 s_v (the mouth) {neck:.4f}")
    results["min_v"] = min_v

    print("\n== [4] k-hat on the funnel, three Gaussian q's, S = 4000, 40 seeds ==")
    rng = np.random.default_rng(43)
    nc_fit_rng = np.random.default_rng(44)
    m_nc, c_nc, _ = fit_gaussian_q(
        lambda zz: noncentred_grad(zz, sigma_v), lambda zz: noncentred_log_p(zz, sigma_v),
        d, nc_fit_rng, True, schedule,
    )
    rows = (
        ("reverse-KL optimum", lambda zz: funnel_log_p(zz, sigma_v), mean_opt, cov_opt),
        ("moment-matched    ", lambda zz: funnel_log_p(zz, sigma_v), *moment_matched(sigma_v, d)),
        ("non-centred fit   ", lambda zz: noncentred_log_p(zz, sigma_v), m_nc, c_nc),
    )
    for label, lp, m, c in rows:
        khat, plain, smooth = importance_summary(rng, lp, m, c, 4000, 40)
        if "non" in label:
            # in (xt, v) the target is N(0, diag(1, 1, sigma_v^2)), so -KL is closed form
            target = np.diag([1.0] * (d - 1) + [sigma_v**2])
            _, ld_t = np.linalg.slogdet(target)
            _, ld_q = np.linalg.slogdet(c)
            elbo = -0.5 * (np.trace(np.linalg.solve(target, c)) + m @ np.linalg.solve(target, m)
                           - d + ld_t - ld_q)
        else:
            elbo = exact_elbo(m, c, sigma_v)
        print(f"  {label}  ELBO {elbo:11.4f}   k-hat {khat.mean():.3f} +- {khat.std():.3f}"
              f"   log Z plain {plain.mean():+.4f} ({plain.std():.4f})"
              f"   PSIS {smooth.mean():+.4f} ({smooth.std():.4f})")
    z_nc = sample(np.random.default_rng(45), 100_000, m_nc, c_nc)
    lw_nc = noncentred_log_p(z_nc, sigma_v) - gaussian_log_q(z_nc, m_nc, c_nc)
    print(f"  non-centred fit: var {np.array2string(np.diag(c_nc), precision=4)} against [1, 1, {sigma_v**2:.0f}]"
          f"   per-draw sd of log w {lw_nc.std():.4f}")

    # the tail each q actually has. for any Gaussian q with finite Var(x), the
    # inner x-integral of E_q[w^a] diverges once exp(-v) < (a-1) / (a S_xx), so
    # E_q[w^a] is infinite for every a > 1: the exact shape is k = 1 for every q
    # in the centred coordinates. printed: how much of q carries the divergence.
    print("  q mass where E_q[w^a | v] is already infinite, reverse-KL optimum / moment-matched")
    for a in (2.0, 1.5, 1.1, 1.01):
        cells = []
        for m, c in ((mean_opt, cov_opt), moment_matched(sigma_v, d)):
            v_star = math.log(a * c[0, 0] / (a - 1))
            cells.append(f"{1 - normal_cdf((v_star - m[-1]) / math.sqrt(c[-1, -1])):.2e}")
        print(f"    a = {a:4.2f}   v* above which it diverges: {math.log(a * cov_opt[0, 0] / (a - 1)):6.3f}"
              f" / {math.log(a * moment_matched(sigma_v, d)[1][0, 0] / (a - 1)):6.3f}"
              f"   q mass {cells[0]} / {cells[1]}")

    print("  reverse-KL optimum, as the sample grows (20 seeds each)")
    for n_draws in (1000, 4000, 16000, 64000):
        khat, plain, smooth = importance_summary(rng, lambda zz: funnel_log_p(zz, sigma_v),
                                                 mean_opt, cov_opt, n_draws, 20)
        print(f"    S={n_draws:6d}  k-hat {khat.mean():.3f} +- {khat.std():.3f}"
              f"   log Z plain {plain.mean():+.4f}   PSIS {smooth.mean():+.4f}")

    # written before the run: k-hat tracks k_v = 1 - s_v^2 / sigma_v^2, the shape
    # day 3's formula gives in the v direction alone, and not the exact k = 1.
    print("\n== [5] sigma_v swept: the closed-form q, k-hat, and MCMC day 4's two columns ==")
    print("  sigma_v  q Var(v)/true  KL      k_v    k-hat (S=4000, 40)   log Z plain   PSIS"
          "      | HMC div  HMC Var ratio")
    sweep = {}
    for sig in (1.0, 2.0, 3.0, 4.0):
        m, c = mean_field_optimum(sig, d)
        ratio = c[-1, -1] / sig**2
        kl = -exact_elbo(m, c, sig)
        k_v = 1.0 - ratio
        rng = np.random.default_rng(int(10 * sig) + 500)
        khat, plain, smooth = importance_summary(rng, lambda zz: funnel_log_p(zz, sig), m, c, 4000, 40)
        sweep[sig] = (ratio, kl, k_v, khat.mean(), plain.mean(), smooth.mean())
        print(f"  {sig:5.1f}    {ratio:.4f}        {kl:.4f}  {k_v:.3f}  {khat.mean():.3f} +- {khat.std():.3f}"
              f"      {plain.mean():+.4f}      {smooth.mean():+.4f}"
              f"   | {MCMC_DIVERGENCES[sig]:6d}   {MCMC_VAR_RATIO[sig]:.3f}")
    results["sweep"] = sweep

    print(f"\ndone in {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
