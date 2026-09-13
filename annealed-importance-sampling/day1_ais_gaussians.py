"""Day 1 - AIS on Gaussian targets, where the path has a closed form at every temperature.

The VI project's gaps were all `log Z - ELBO` against a normaliser written down
in advance, and the only estimate of `log Z` it built came from q. AIS estimates
Z along a path out of q. With a Gaussian start and a Gaussian target every
density on the geometric path is Gaussian, so for exact transitions the bias of
`log Z`, the variance of the log weight and the second moment of the weight are
closed forms, and the simulated sampler is scored against them before anything
is read off it. The target is VI day 3's AR(1), r = 0.9, d = 8, with
`log Z = 1.538949`. Two starts: the mean-field CAVI optimum, KL 2.610346, and
N(0, I), KL 25.946839.

1. At one temperature AIS is the ELBO. `E[log w] = ELBO` exactly, so the bias
   of `log Z` at T = 1 is `KL(q || p)`, and with no transitions it stays the
   ELBO at every T, asserted on the same draws at T = 1, 7 and 1000. For exact
   transitions the bias is also the sum of the KLs between neighbouring
   temperatures, and the two derivations agree to 5.5e-14. The simulated mean
   log weight sits inside 2 standard errors of the closed form on all ten rows.

2. The ELBO's head start does not survive annealing. With exact transitions the
   bias falls like `J / 2T`, where J is the integral of `Var_beta(g)` along the
   path, and J is 29.84 from the CAVI start against 33.47 from N(0, I). The CAVI
   start is 10x closer in KL and ends up 2.3x less biased at T = 10, 1.24x at
   T = 100 and 1.13x at T = 1000. The precision interpolates linearly, so along
   the target's long axis the path's variance is 0.12 at beta = 0, 1.0 at
   beta = 0.9 and 6.2 at beta = 1, and 79% of J sits in the last tenth. From
   N(0, I), 51% of it sits in the first.

3. The second moment of the weight is a product of normalisers at temperatures
   past the path, since `f_t^2 / f_{t-1}` is f at `2 beta_t - beta_{t-1}`. From
   the CAVI start the path stops being normalisable at beta* = 1.019, so
   `E[w^2]` is infinite until T > 52.63. N(0, I) has beta* = 1.192 and a finite
   second moment from T = 6. Where it is infinite the unbiased estimate of Z
   reads low: `log mean w` is 0.77 below `log Z` over 40000 runs at T = 1. At
   T = 30 the bias of the log of an N-run mean goes 0.386, 0.107, 0.024 at
   N = 1, 10, 100, which is 3.6x and then 4.4x per decade, and 64% of the
   N = 100 estimates fall below `log Z` against 52-53% in the three cases with a
   finite second moment. N = 1000 gives 0.003 +- 0.006, which does not resolve.

4. Real transitions cost far more than the path does. From the CAVI start, MALA
   at 94% acceptance with one step per temperature is 8.2x the exact-transition
   bias at T = 100 and 35x at T = 1000, and ten steps are 3.7x and 11x. The
   chains lag along the long axis, 3.74 of 5.89 after 10000 steps, against 5.44
   of 6.17 from N(0, I), which starts at 1.0 there instead of 0.12. From the
   CAVI start the split of a fixed budget barely matters, 1.23 against 1.13 at
   100 gradient steps and 0.50 against 0.52 at 1000. From N(0, I) temperatures
   beat steps, 2.82 against 1.82 and 0.42 against 0.31. And the head start
   reverses between 100 and 1000 gradient steps. At 10000 the CAVI start is
   0.162 nats low against 0.043 from N(0, I).

Five predictions written before the run, and three were wrong.

- Right: the two derivations agree and the simulation matches them, and MALA
  with one step per temperature is at least 2x the exact-transition bias at
  T = 100. It is 8.2x and 10.7x.
- Wrong: T * bias falls monotonically from KL(q || p) at T = 1 to J / 2. It does
  from N(0, I), 25.95 down to 16.77. From the CAVI start it rises, 2.61 up to
  14.80, because KL is small there and J is not.
- Wrong: `Var(log w) / (2 bias)` within 10% of 1 from T = 10. At T = 10 it is
  0.54 from the CAVI start and 1.54 from N(0, I). It gets inside 10% only from
  T = 300 and T = 100, and from opposite sides, since the higher cumulants of
  log w add to the bias from one start and take away from it from the other.
- Wrong: the delta method for the log of an N-run mean fine while
  Var(log w) <= 1 and underestimating from 4. The case that breaks it has
  Var(log w) = 0.55. What decides it is `E[w^2]`, which Var(log w) does not see,
  and where that is infinite the delta method puts the bias at infinity and it
  is 0.024 at N = 100.

NumPy only. Fixed seeds. About 70 seconds.
"""

import time

import numpy as np

RNG = np.random.default_rng(1)


def ar1(d: int, r: float) -> np.ndarray:
    """Unit-variance covariance with corr(i, j) = r ** |i - j|."""
    idx = np.arange(d)
    return r ** np.abs(idx[:, None] - idx[None, :])


# ----------------------------------------------------------------------------
# The geometric path between a normalised Gaussian q and an unnormalised
# Gaussian target, in closed form at every beta
# ----------------------------------------------------------------------------


class GeometricPath:
    """f_beta(x) = q(x)^(1 - beta) * p~(x)^beta for Gaussian q and p~.

    Every f_beta is an unnormalised Gaussian with precision
    `(1 - beta) Lambda_0 + beta Lambda`, so its normaliser, its mean and the
    moments of `g = log p~ - log q` under it are all closed forms, and the
    formulas hold for any beta at which that precision is positive definite -
    including beta outside [0, 1], which is where the second moment of the
    weights turns out to live.
    """

    def __init__(self, q_mean, q_cov, mean, cov):
        self.d = len(mean)
        self.m0 = np.asarray(q_mean, dtype=float)
        self.prec0 = np.linalg.inv(q_cov)
        self.mu = np.asarray(mean, dtype=float)
        self.prec = np.linalg.inv(cov)
        self.log_z0 = 0.5 * (self.d * np.log(2.0 * np.pi) + np.linalg.slogdet(q_cov)[1])
        self.a = self.prec - self.prec0
        self.b = self.prec @ self.mu - self.prec0 @ self.m0
        self.g0 = -0.5 * self.mu @ self.prec @ self.mu + 0.5 * self.m0 @ self.prec0 @ self.m0 + self.log_z0

    def log_z(self) -> float:
        return 0.5 * (self.d * np.log(2.0 * np.pi) - np.linalg.slogdet(self.prec)[1])

    def at(self, beta: float):
        """(mean, cov, precision, log Z_beta), or None where f_beta has no normaliser."""
        p = (1.0 - beta) * self.prec0 + beta * self.prec
        if np.linalg.eigvalsh(p)[0] <= 0.0:
            return None
        h = (1.0 - beta) * self.prec0 @ self.m0 + beta * self.prec @ self.mu
        m = np.linalg.solve(p, h)
        c = (1.0 - beta) * self.m0 @ self.prec0 @ self.m0 + beta * self.mu @ self.prec @ self.mu - m @ p @ m
        log_z = (-(1.0 - beta) * self.log_z0 + 0.5 * self.d * np.log(2.0 * np.pi)
                 - 0.5 * np.linalg.slogdet(p)[1] - 0.5 * c)
        return m, np.linalg.inv(p), p, log_z

    def beta_star(self) -> float:
        """The largest beta the path can be extended to and stay normalisable."""
        w, v = np.linalg.eigh(self.prec0)
        root_inv = v @ np.diag(w ** -0.5) @ v.T
        lam = np.linalg.eigvalsh(root_inv @ self.a @ root_inv)[0]
        return np.inf if lam >= 0.0 else -1.0 / lam

    def g(self, x: np.ndarray) -> np.ndarray:
        """log p~(x) - log q(x), row-wise."""
        return -0.5 * np.einsum("ni,ij,nj->n", x, self.a, x) + x @ self.b + self.g0

    def g_moments(self, beta: float):
        """E and Var of g under pi_beta. g is a quadratic form, so both are exact."""
        m, s, _, _ = self.at(beta)
        mean = -0.5 * (np.trace(self.a @ s) + m @ self.a @ m) + self.b @ m + self.g0
        v = self.b - self.a @ m
        var = 0.5 * np.trace(self.a @ s @ self.a @ s) + v @ s @ v
        return mean, var

    def kl(self, beta1: float, beta2: float) -> float:
        m1, s1, _, _ = self.at(beta1)
        m2, s2, p2, _ = self.at(beta2)
        dm = m2 - m1
        return 0.5 * (np.trace(p2 @ s1) + dm @ p2 @ dm - self.d
                      + np.linalg.slogdet(s2)[1] - np.linalg.slogdet(s1)[1])


def exact_log_weight(path: GeometricPath, betas: np.ndarray):
    """For exact transitions: E[log w] two ways, Var[log w], and log E[w^2]/Z^2.

    With x_{t-1} drawn exactly from pi_{t-1}, the increments
    `(beta_t - beta_{t-1}) g(x_{t-1})` are independent, so the mean and variance
    are sums. The bias has a second derivation: each increment's expectation is
    `log Z_t - log Z_{t-1} - KL(pi_{t-1} || pi_t)`, so the bias is the sum of
    the KLs. And the square of an increment is
    `f_t^2 / f_{t-1}^2 = f_{2 beta_t - beta_{t-1}} / f_{t-1}`, one step past
    beta_t along the same line, so the second moment is a product of normalisers
    at extrapolated temperatures and is infinite when any of them is.
    """
    mean = var = kl_sum = 0.0
    log_second = 0.0
    for t in range(1, len(betas)):
        step = betas[t] - betas[t - 1]
        e, v = path.g_moments(betas[t - 1])
        mean += step * e
        var += step * step * v
        kl_sum += path.kl(betas[t - 1], betas[t])
        ext = path.at(2.0 * betas[t] - betas[t - 1])
        if ext is None or not np.isfinite(log_second):
            log_second = np.inf
        else:
            log_second += ext[3] + path.at(betas[t - 1])[3] - 2.0 * path.at(betas[t])[3]
    return path.log_z() - mean, kl_sum, var, log_second


def thermodynamic_length(path: GeometricPath, points: int = 20001):
    """J = integral over [0, 1] of Var_beta(g). Uniform-schedule AIS has bias -> J / 2T.

    Also returns the shares of J in beta < 0.1 and beta > 0.9.
    """
    grid = np.linspace(0.0, 1.0, points)
    var = np.array([path.g_moments(beta)[1] for beta in grid])
    j = float(np.trapezoid(var, grid))
    low = float(np.trapezoid(var[grid <= 0.1], grid[grid <= 0.1])) / j
    high = float(np.trapezoid(var[grid >= 0.9], grid[grid >= 0.9])) / j
    return j, low, high


# ----------------------------------------------------------------------------
# AIS, simulated
# ----------------------------------------------------------------------------


def ais_exact(path: GeometricPath, betas: np.ndarray, runs: int, chunk: int = 4000) -> np.ndarray:
    """AIS whose transition at each temperature is an exact draw from pi_beta."""
    stats = [path.at(beta) for beta in betas[:-1]]
    chols = [np.linalg.cholesky(s) for _, s, _, _ in stats]
    out = []
    for start in range(0, runs, chunk):
        n = min(chunk, runs - start)
        log_w = np.zeros(n)
        for t in range(1, len(betas)):
            m, _, _, _ = stats[t - 1]
            x = m + RNG.standard_normal((n, path.d)) @ chols[t - 1].T
            log_w += (betas[t] - betas[t - 1]) * path.g(x)
        out.append(log_w)
    return np.concatenate(out)


def ais_mala(path: GeometricPath, betas: np.ndarray, runs: int, steps: int, scale: float = 0.5):
    """AIS with `steps` MALA moves per temperature, all runs in parallel.

    The step size at each temperature is `scale` times pi_beta's smallest
    variance, which is the sampler being told the scale of its target - more
    than a real run gets, and it still has to cross the long direction by
    diffusion. Returns `(log_w, acceptance rate, final states)`, the final
    states being the ones the last weight increment was computed at.
    """
    m0, s0, _, _ = path.at(0.0)
    x = m0 + RNG.standard_normal((runs, path.d)) @ np.linalg.cholesky(s0).T
    log_w = np.zeros(runs)
    accepted = proposed = 0
    for t in range(1, len(betas)):
        log_w += (betas[t] - betas[t - 1]) * path.g(x)
        if t == len(betas) - 1:
            break
        m, _, p, _ = path.at(betas[t])
        h = scale / np.linalg.eigvalsh(p)[-1]
        for _ in range(steps):
            grad = -(x - m) @ p
            y = x + 0.5 * h * grad + np.sqrt(h) * RNG.standard_normal(x.shape)
            grad_y = -(y - m) @ p
            log_target = (-0.5 * np.einsum("ni,ij,nj->n", y - m, p, y - m)
                          + 0.5 * np.einsum("ni,ij,nj->n", x - m, p, x - m))
            log_forward = -np.sum((y - x - 0.5 * h * grad) ** 2, axis=1) / (2 * h)
            log_backward = -np.sum((x - y - 0.5 * h * grad_y) ** 2, axis=1) / (2 * h)
            accept = np.log(RNG.random(runs)) < log_target + log_backward - log_forward
            x[accept] = y[accept]
            accepted += int(accept.sum())
            proposed += runs
    return log_w, accepted / proposed, x


def log_mean_exp(a: np.ndarray, axis=None):
    top = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.mean(np.exp(a - top), axis=axis))


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
    log_z = starts["cavi q"].log_z()
    print(f"target: AR(1) r = 0.9, d = {d}, mean linspace(-1, 1)   log Z = {log_z:.6f}")

    print("\n== the bias two ways, and at T = 1 it is the ELBO gap ==")
    for name, path in starts.items():
        e1, v1 = path.g_moments(0.0)
        m0, s0 = path.m0, np.linalg.inv(path.prec0)
        dm = path.mu - m0
        kl_q_p = 0.5 * (np.trace(path.prec @ s0) + dm @ path.prec @ dm - d
                        + np.linalg.slogdet(cov)[1] - np.linalg.slogdet(s0)[1])
        worst = 0.0
        for T in (1, 10, 100, 1000):
            by_moments, by_kl, _, _ = exact_log_weight(path, np.linspace(0.0, 1.0, T + 1))
            worst = max(worst, abs(by_moments - by_kl))
            if T == 1:
                assert abs(by_kl - kl_q_p) < 1e-10 and abs(log_z - e1 - kl_q_p) < 1e-10
        print(f"  {name:8s} ELBO {e1:.6f}   log Z - ELBO {log_z - e1:.6f}   KL(q||p) {kl_q_p:.6f}"
              f"   max |sum KL - (log Z - E log w)| over T {worst:.1e}")
        assert worst < 1e-9

    print("\n== annealing with no transitions is the ELBO at every T ==")
    path = starts["cavi q"]
    x0 = path.m0 + RNG.standard_normal((5, d)) * np.sqrt(np.diag(np.linalg.inv(path.prec0)))
    for T in (1, 7, 1000):
        betas = np.linspace(0.0, 1.0, T + 1)
        log_w = sum((betas[t] - betas[t - 1]) * path.g(x0) for t in range(1, T + 1))
        print(f"  T={T:5d}  log w = {np.array2string(log_w, precision=12)}")
        assert np.allclose(log_w, path.g(x0), atol=1e-10)

    print("\n== exact transitions, closed form ==")
    for name, path in starts.items():
        j, low, high = thermodynamic_length(path)
        b_star = path.beta_star()
        print(f"  {name}: J = {j:.4f}   J/2 = {j / 2:.4f}   share of J in beta < 0.1 {low:.4f}, in beta > 0.9 {high:.4f}")
        print(f"      beta* = {b_star:.6f}   E[w^2] finite iff T > {1.0 / (b_star - 1.0) if np.isfinite(b_star) else 0.0:.2f}")
        print("      T      bias    T*bias   Var(log w)   Var/(2 bias)   Var(w)/Z^2   exp(Var)-1")
        for T in (1, 2, 3, 5, 10, 30, 100, 300, 1000):
            bias, _, var, log_second = exact_log_weight(path, np.linspace(0.0, 1.0, T + 1))
            rel = np.expm1(log_second) if np.isfinite(log_second) else np.inf
            print(f"  {T:5d}  {bias:8.5f}  {T * bias:8.4f}  {var:10.5f}  {var / (2 * bias):12.4f}"
                  f"   {rel:10.4g}   {np.expm1(min(var, 700.0)):10.4g}")

    print("\n== exact transitions, simulated ==")
    for name, path in starts.items():
        for T, runs in ((1, 40000), (10, 40000), (30, 40000), (100, 20000), (1000, 4000)):
            betas = np.linspace(0.0, 1.0, T + 1)
            bias, _, var, log_second = exact_log_weight(path, betas)
            log_w = ais_exact(path, betas, runs)
            se = log_w.std(ddof=1) / np.sqrt(runs)
            z_hat = log_mean_exp(log_w) - log_z
            print(f"  {name:8s} T={T:5d} runs={runs:6d}  E log w - log Z {log_w.mean() - log_z:+.4f} +- {se:.4f}"
                  f" (exact {-bias:+.4f})   Var {log_w.var(ddof=1):8.4f} (exact {var:8.4f})"
                  f"   log mean w - log Z {z_hat:+.4f}   E[w^2] {'finite' if np.isfinite(log_second) else 'infinite'}")

    print("\n== log of the mean of N weights, exact transitions ==")
    for name, T in (("cavi q", 30), ("cavi q", 100), ("N(0, I)", 30), ("N(0, I)", 100)):
        path = starts[name]
        betas = np.linspace(0.0, 1.0, T + 1)
        bias, _, var, log_second = exact_log_weight(path, betas)
        rel = np.expm1(log_second) if np.isfinite(log_second) else np.inf
        repeats = 2000
        log_w = ais_exact(path, betas, repeats * 100).reshape(repeats, 100)
        for n in (1, 10, 100):
            est = log_mean_exp(log_w[:, :n], axis=1) - log_z
            print(f"  {name:8s} T={T:4d}  N={n:4d}  E[log Z_N] - log Z {est.mean():+.4f} +- {est.std(ddof=1) / np.sqrt(repeats):.4f}"
                  f"   delta -Var(w)/2NZ^2 {-rel / (2 * n):+.4f}   below log Z {np.mean(est < 0):.3f}")
    # one more decade where the second moment is infinite, at fewer repeats.
    path = starts["cavi q"]
    betas = np.linspace(0.0, 1.0, 31)
    repeats = 400
    log_w = ais_exact(path, betas, repeats * 1000).reshape(repeats, 1000)
    est = log_mean_exp(log_w, axis=1) - log_z
    print(f"  cavi q   T=  30  N=1000  E[log Z_N] - log Z {est.mean():+.4f} +- {est.std(ddof=1) / np.sqrt(repeats):.4f}"
          f"   ({repeats} repeats)   below log Z {np.mean(est < 0):.3f}")

    print("\n== MALA transitions ==")
    long_axis = np.linalg.eigh(cov)[1][:, -1]
    print(f"  target variance along its longest axis {long_axis @ cov @ long_axis:.3f}"
          f"   cavi q's {long_axis @ np.linalg.inv(starts['cavi q'].prec0) @ long_axis:.3f}   N(0, I)'s 1.000")
    for name, path in starts.items():
        for T in (10, 100, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            bias, _, var, _ = exact_log_weight(path, betas)
            _, s_last, _, _ = path.at(betas[-2])
            for steps in (1, 10):
                runs = 2000
                log_w, acc, x = ais_mala(path, betas, runs, steps)
                se = log_w.std(ddof=1) / np.sqrt(runs)
                print(f"  {name:8s} T={T:5d} steps={steps:2d}  bias {log_z - log_w.mean():7.4f} +- {se:.4f}"
                      f"   exact-transition bias {bias:7.4f}   ratio {(log_z - log_w.mean()) / bias:6.2f}"
                      f"   Var(log w) {log_w.var(ddof=1):8.4f}   accept {acc:.3f}"
                      f"   final long-axis var {np.var(x @ long_axis):6.3f} of {long_axis @ s_last @ long_axis:.3f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
