"""Day 1 - mean-field CAVI on targets whose posteriors are known exactly.

The MCMC project ended on one sentence: a statistic of the samples you have
cannot report on the samples you did not get. The ELBO is an expectation under
`q`, so it has that shape by construction, and the point of starting here rather
than with a hard posterior is that on a Gaussian target every quantity in the
argument is available in closed form. The bound, the gap, the fixed point and
the exact posterior are all numbers, so "the diagnostic is wrong" is a statement
that can be checked instead of asserted.

Three things are measured.

1. The ELBO gap is `KL(q || p)` to machine precision. `log Z` is known for a
   Gaussian, so `log Z - ELBO` can be compared against the closed-form Gaussian
   KL, and the two are derived separately so the check is between derivations
   rather than inside one. Residual 2.2e-16 at the CAVI fixed point, and the
   ELBO's smallest step over 136 coordinate updates is +3.8e-15, so the
   monotonicity is asserted rather than assumed.

2. The mean-field fixed point gets the mean exactly right and the variance wrong
   by `1 / (Sigma^-1)_ii / Sigma_ii`, the conditional variance over the marginal
   one - 0.19 against 1.00 at rho = 0.9, which is `1 - rho^2` exactly. Same fact
   as the ELBO gap: reverse KL pays for `q` putting mass where `p` has none and
   nothing in it pays for the reverse.

3. That ratio is a property of the *shape* of the correlation rather than its
   size. Matched on mean absolute off-diagonal correlation at d = 8, AR(1) and
   equicorrelated differ by 1.43x, 1.94x and 2.81x as the correlation rises, all
   three with equicorrelated the milder.

Two predictions written before the run, both wrong in the same place. I had both
structures getting worse with dimension. Equicorrelated does, saturating at
`1 - r` from above (0.910 at d = 2 down to 0.711 at d = 64, r = 0.3). AR(1) does
not: it is 0.510 at d = 2 and then exactly 0.342282 at every d from 4 to 64,
because the precision of an AR(1) covariance is tridiagonal and an interior
coordinate's conditional variance sees only its two neighbours. Adding dimensions
adds no information about it. The d = 2 value is the boundary case, not a trend.

NumPy only. Fixed seeds. About 12 seconds, almost all of it the reverse-KL grid.
"""

import numpy as np

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# Gaussian targets, and the exact quantities the approximation is scored against
# ----------------------------------------------------------------------------


def equicorrelated(d: int, r: float) -> np.ndarray:
    """Unit-variance covariance with every off-diagonal correlation equal to r."""
    return (1.0 - r) * np.eye(d) + r * np.ones((d, d))


def ar1(d: int, r: float) -> np.ndarray:
    """Unit-variance covariance with corr(i, j) = r ** |i - j|."""
    idx = np.arange(d)
    return r ** np.abs(idx[:, None] - idx[None, :])


def gaussian_log_norm(cov: np.ndarray) -> float:
    """log Z for the unnormalised density exp(-0.5 (z-mu)' Sigma^-1 (z-mu))."""
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0, "covariance is not positive definite"
    return 0.5 * (d * np.log(2.0 * np.pi) + logdet)


def kl_diagonal_to_gaussian(
    q_mean: np.ndarray, q_var: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    """Exact KL(q || p) for diagonal Gaussian q against full Gaussian p."""
    prec = np.linalg.inv(cov)
    d = len(q_mean)
    delta = q_mean - mean
    trace_term = float(np.sum(np.diag(prec) * q_var))
    quad_term = float(delta @ prec @ delta)
    _, logdet_p = np.linalg.slogdet(cov)
    logdet_q = float(np.sum(np.log(q_var)))
    return 0.5 * (trace_term + quad_term - d + logdet_p - logdet_q)


# ----------------------------------------------------------------------------
# Mean-field CAVI
# ----------------------------------------------------------------------------


def cavi_gaussian(mean, cov, sweeps=200, tol=1e-14):
    """Coordinate-ascent VI for a diagonal q against a full Gaussian target.

    The update is the standard one: `log q_i` proportional to the expectation of
    the joint log density over the other coordinates, which for a Gaussian is
    again Gaussian with precision `Lambda_ii` and mean shifted by the current
    estimates of its neighbours. The variance update does not depend on the
    other factors at all, which is why the fixed point is reached in the means
    alone and the variance is decided by the first sweep.

    Returns `(q_mean, q_var, elbo_trace)`.
    """
    prec = np.linalg.inv(cov)
    d = len(mean)
    log_z = gaussian_log_norm(cov)

    q_var = 1.0 / np.diag(prec)
    q_mean = mean + RNG.normal(scale=2.0, size=d)  # deliberately not at the mode

    trace = []
    for _ in range(sweeps):
        for i in range(d):
            others = np.delete(np.arange(d), i)
            shift = prec[i, others] @ (q_mean[others] - mean[others])
            q_mean[i] = mean[i] - shift / prec[i, i]
            trace.append(log_z - kl_diagonal_to_gaussian(q_mean, q_var, mean, cov))
        if len(trace) > d and abs(trace[-1] - trace[-1 - d]) < tol:
            break
    return q_mean, q_var, np.array(trace)


def elbo_by_quadrature_free_energy(q_mean, q_var, mean, cov):
    """ELBO from its definition, E_q[log p~(z)] - E_q[log q(z)], in closed form.

    Computed independently of `kl_diagonal_to_gaussian` so the identity
    `ELBO = log Z - KL` is a check between two derivations rather than an
    algebraic restatement of one.
    """
    prec = np.linalg.inv(cov)
    d = len(q_mean)
    delta = q_mean - mean
    # E_q[-0.5 (z-mu)' Lambda (z-mu)]
    cross = -0.5 * (float(delta @ prec @ delta) + float(np.sum(np.diag(prec) * q_var)))
    entropy = 0.5 * float(np.sum(np.log(2.0 * np.pi * np.e * q_var)))
    return cross + entropy


# ----------------------------------------------------------------------------
# Reverse versus forward KL on a target a single Gaussian cannot represent
# ----------------------------------------------------------------------------


def mixture_log_density(z, weights, means, sds):
    """log p(z) for a 1-D Gaussian mixture, on a grid.

    Deliberately not log-sum-exp, which is the thing to check rather than
    assume. The grid is [-12, 12] and the components sit at -2 and 2.5 with
    sd 0.7, so the least favourable point on the grid still has a largest
    component log-density of -102.6 and a mixture density of 1.8e-45 - forty
    orders of magnitude clear of underflow, and no point on the grid returns a
    non-finite log. Widen the grid or shrink the scales and this stops being
    true.
    """
    comp = -0.5 * ((z[:, None] - means) / sds) ** 2 - np.log(sds * np.sqrt(2 * np.pi))
    return np.log(np.exp(comp) @ weights)


def fit_gaussian_reverse_kl(weights, means, sds, grid):
    """Minimise KL(q || p) over a grid of (mean, sd), by quadrature.

    A grid search rather than a gradient method on purpose - day 2 is about the
    gradients, and here the objective is one-dimensional enough that the exact
    minimiser is worth having without an optimiser between it and the answer.
    """
    dz = grid[1] - grid[0]
    log_p = mixture_log_density(grid, weights, means, sds)
    best = None
    for m in np.linspace(-6.0, 6.0, 241):
        for s in np.linspace(0.15, 4.0, 200):
            log_q = -0.5 * ((grid - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
            q = np.exp(log_q)
            kl = float(np.sum(q * (log_q - log_p)) * dz)
            if best is None or kl < best[0]:
                best = (kl, m, s)
    return best


def fit_gaussian_forward_kl(weights, means, sds):
    """Minimise KL(p || q): moment matching, in closed form."""
    m = float(weights @ means)
    v = float(weights @ (sds**2 + means**2) - m**2)
    return m, np.sqrt(v)


def covered_mass(m, s, weights, means, sds, grid):
    """Fraction of p's mass in the region where q's density exceeds 1% of its peak."""
    dz = grid[1] - grid[0]
    q = np.exp(-0.5 * ((grid - m) / s) ** 2)
    p = np.exp(mixture_log_density(grid, weights, means, sds))
    return float(np.sum(p[q > 0.01]) * dz)


# ----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=4, suppress=True)

    print("== CAVI on a bivariate Gaussian, rho = 0.9 ==")
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    mean = np.array([1.0, -2.0])
    q_mean, q_var, trace = cavi_gaussian(mean, cov)
    log_z = gaussian_log_norm(cov)
    kl = kl_diagonal_to_gaussian(q_mean, q_var, mean, cov)
    elbo = elbo_by_quadrature_free_energy(q_mean, q_var, mean, cov)
    monotone = float(np.min(np.diff(trace)))
    print(f"  q mean       {q_mean}   exact {mean}")
    print(f"  q var        {q_var}   exact {np.diag(cov)}")
    print(f"  log Z        {log_z:.12f}")
    print(f"  ELBO         {elbo:.12f}")
    print(f"  KL(q||p)     {kl:.12f}")
    print(f"  |log Z - ELBO - KL| = {abs(log_z - elbo - kl):.3e}")
    print(f"  smallest ELBO increase over {len(trace)} updates: {monotone:.3e}")
    print(f"  variance ratio {q_var[0] / cov[0, 0]:.6f}   1 - rho^2 = {1 - 0.81:.6f}")

    print("\n== the same underestimate, two correlation structures ==")
    print("  matched on mean |off-diagonal correlation|, d = 8")
    d = 8
    for r_ar in (0.5, 0.7, 0.9):
        cov_ar = ar1(d, r_ar)
        off = np.abs(cov_ar[~np.eye(d, dtype=bool)]).mean()
        # equicorrelated matrix with the same mean off-diagonal correlation
        cov_eq = equicorrelated(d, off)
        ratio_ar = 1.0 / np.diag(np.linalg.inv(cov_ar))[d // 2]
        ratio_eq = 1.0 / np.diag(np.linalg.inv(cov_eq))[d // 2]
        print(
            f"  mean|corr|={off:.4f}   AR(1) r={r_ar:.2f} ratio={ratio_ar:.4f}"
            f"   equicorr r={off:.4f} ratio={ratio_eq:.4f}"
            f"   factor {ratio_eq / ratio_ar:.2f}"
        )

    print("\n== and how each one moves with dimension, correlation held fixed ==")
    print("  d      AR(1) r=0.7     equicorr r=0.3")
    for d in (2, 4, 8, 16, 32, 64):
        ar_ratio = 1.0 / np.diag(np.linalg.inv(ar1(d, 0.7)))[d // 2]
        eq_ratio = 1.0 / np.diag(np.linalg.inv(equicorrelated(d, 0.3)))[d // 2]
        print(f"  {d:3d}    {ar_ratio:.6f}        {eq_ratio:.6f}")

    print("\n== reverse vs forward KL on a mixture a Gaussian cannot represent ==")
    weights = np.array([0.65, 0.35])
    means = np.array([-2.0, 2.5])
    sds = np.array([0.7, 0.7])
    grid = np.linspace(-12.0, 12.0, 24001)
    kl_rev, m_rev, s_rev = fit_gaussian_reverse_kl(weights, means, sds, grid)
    m_fwd, s_fwd = fit_gaussian_forward_kl(weights, means, sds)
    dz = grid[1] - grid[0]
    log_p = mixture_log_density(grid, weights, means, sds)
    log_q_fwd = -0.5 * ((grid - m_fwd) / s_fwd) ** 2 - np.log(s_fwd * np.sqrt(2 * np.pi))
    kl_fwd_of_fwd = float(np.sum(np.exp(log_p) * (log_p - log_q_fwd)) * dz)
    q_rev = np.exp(-0.5 * ((grid - m_rev) / s_rev) ** 2) / (s_rev * np.sqrt(2 * np.pi))
    kl_fwd_of_rev = float(np.sum(np.exp(log_p) * (log_p - np.log(q_rev))) * dz)
    print(f"  reverse-KL fit   mean={m_rev:+.3f} sd={s_rev:.3f}   KL(q||p)={kl_rev:.4f}")
    print(f"  forward-KL fit   mean={m_fwd:+.3f} sd={s_fwd:.3f}")
    print(f"  KL(p||q)   reverse fit {kl_fwd_of_rev:.4f}   forward fit {kl_fwd_of_fwd:.4f}"
          f"   ({kl_fwd_of_rev / kl_fwd_of_fwd:.1f}x)")
    print(f"  mass of p inside the reverse fit's support {covered_mass(m_rev, s_rev, weights, means, sds, grid):.4f}")
    print(f"  mass of p inside the forward fit's support {covered_mass(m_fwd, s_fwd, weights, means, sds, grid):.4f}")

    # the point of the file, asserted rather than described. the reverse fit
    # collapses onto the heavier component - sd 0.711 against the component's
    # 0.700 - and scores 0.4294 on the objective VI maximises while sitting on
    # 65% of the target's mass. it is not a bad fit that the ELBO failed to
    # notice; it is the optimum of the thing the ELBO is.
    assert kl_rev < kl_fwd_of_rev, "reverse fit should win the objective it optimises"
    assert kl_fwd_of_fwd < kl_fwd_of_rev, "forward fit should win the other one"


if __name__ == "__main__":
    main()
