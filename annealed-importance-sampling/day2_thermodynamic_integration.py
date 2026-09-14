"""Day 2 - thermodynamic integration on the same paths, and what the path and the schedule cost.

TI writes `log Z` as the integral over beta of `E_beta[d log f / d beta]`, which on
the geometric path is `E_beta[g]`. Day 1's closed forms give that integrand at
every beta, so a quadrature rule and a schedule can be scored against the exact
integral with no sampling in the way. The second path interpolates moments
instead of natural parameters (Grosse, Maddison and Salakhutdinov 2013). Both are
written as one curve of unnormalised Gaussians, and the geometric one matches
day 1's closed forms to 3.6e-15. Same AR(1) target, same two starts.

1. AIS with exact transitions is TI's left Riemann sum. `E[log w]` is
   `sum step * E_{beta_{t-1}}[g]`, asserted to 1e-10 at every T, and the right sum
   sits above `log Z` by exactly `(E_1[g] - E_0[g]) / T`. So day 1's bias is a
   quadrature error, and the trapezoid rule removes its first-order part. At
   T = 100 it is 0.0114 against AIS's 0.1378 from the CAVI start and 0.0032
   against 0.1705 from N(0, I). The noise stays. With one exact draw at each
   temperature the trapezoid estimate has sd 0.524 against AIS's 0.490 at
   T = 100, and 0.1718 against 0.1707 at T = 1000. From the CAVI start at T = 1 it
   is 18.8 against 1.46, because the trapezoid puts half its weight on beta = 1,
   where Var(g) is 1409.5.

2. J is not a property of the path. On the geometric path it is
   `E_1[g] - E_0[g]`, which is `KL(q || p) + KL(p || q)`, and the moment-averaged
   path gives the same 29.8421 and 33.4737, asserted to 1e-6. From the CAVI start,
   where the means agree, the two paths are mirror images. Whitened by q, one
   interpolates precisions and the other covariances with the same eigenvectors,
   so a KL step from a to b on one is the step from 1 - b to 1 - a on the other,
   asserted to 5.3e-15. 79% of J sits in the last tenth of the geometric path and
   79% in the first tenth of the other, and the exact-transition bias is identical
   at every T.

   That corrects day 1. It put the CAVI start's lost head start down to the path
   holding q narrow along the long axis until the last tenth. The mirrored path
   widens it at once and loses exactly as much. What decides it is the reverse KL.
   The CAVI q is 10x closer in KL(q || p), 2.61 against 25.95, and 3.6x further in
   KL(p || q), 27.23 against 7.53, and the large-T bias only sees the sum.

3. A schedule buys at most J / L^2, L the integral of sqrt(Var). That is 2.454 on
   either path from the CAVI start and 1.37 to 1.39 from N(0, I), and the exact
   gains at T = 1000 are 2.437 and 1.370 on the geometric path. The adaptive
   schedule puts 39% of the temperatures past beta = 0.9 there. For TI the same
   schedule cuts the trapezoid error 17x at T = 100 from the CAVI start.

4. With MALA, the two paths exact transitions cannot tell apart are 2x apart. At
   T = 1000 with one step per temperature the geometric path is 0.534 nats low from
   the CAVI start and the moment-averaged path 0.258, and 0.309 against 0.131 from
   N(0, I). Day 1's ratios reproduce on a new seed, 8.16x and 36.1x the
   exact-transition bias at T = 100 and 1000. Along the long axis the
   moment-averaged path from the CAVI start has variance 1.64 at beta = 0.25
   against 0.16 on the geometric one, so its chains widen over the whole schedule
   and not in the last tenth. The adaptive schedule is worth less under MALA than
   with exact transitions, 1.67x against 2.44x on the geometric path from the CAVI
   start and nothing at T = 100. From N(0, I) it makes the geometric path worse,
   0.361 against 0.309, 2.2 standard errors, and I have no mechanism for that I
   have checked.

Six predictions written before the run. Three right, two wrong, one half.

- Right: the left-sum identity. Under MALA the adaptive schedule gains less than
  its exact-transition gain at T = 1000 from the CAVI start. The trapezoid error
  is under a tenth of AIS's bias at T = 100 from both starts, 12x and 54x.
- Wrong: the moment-averaged path has J under half the geometric path's from the
  CAVI start and more than it from N(0, I). It is equal both times, and it is an
  identity.
- Wrong: the adaptive gain on the geometric path is at least 3 from the CAVI
  start. It is 2.454. The other half, under 2 from N(0, I), held at 1.369.
- Half: the trapezoid error within 10% of `(Var_1 - Var_0) / 12T^2` only from
  T = 100 from the CAVI start held, at 0.558, 0.841 and 0.976 of it for
  T = 10, 30 and 100. From T = 10 from N(0, I) did not. It is 0.871 there and
  inside 10% from T = 30.

NumPy only. Fixed seeds. About 45 seconds.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day1_ais_gaussians import GeometricPath, ar1, exact_log_weight  # noqa: E402

RNG = np.random.default_rng(2)
LOG_2PI = np.log(2.0 * np.pi)


# ----------------------------------------------------------------------------
# Any curve of unnormalised Gaussians, and the two curves compared here
# ----------------------------------------------------------------------------


class GaussianPath:
    """log f_beta(x) = -x' P x / 2 + h' x + c, with (P, h, c) any smooth curve in beta.

    `curve(beta)` gives (P, h, c) and `slope(beta)` their derivatives. Everything
    else is read off those two: pi_beta's mean and covariance, log Z_beta, the KL
    between two temperatures, and the mean and variance under pi_beta of
    `d log f / d beta`, which is what TI integrates and what the thermodynamic
    length is built from. Day 1's path is the special case where all three are
    linear in beta and the slope is g.
    """

    def __init__(self, name, curve, slope, d):
        self.name = name
        self.curve = curve
        self.slope = slope
        self.d = d

    def at(self, beta: float):
        p, h, c = self.curve(beta)
        s = np.linalg.inv(p)
        m = s @ h
        log_z = c + 0.5 * (self.d * LOG_2PI - np.linalg.slogdet(p)[1] + h @ m)
        return m, s, p, log_z

    def log_f(self, beta: float, x: np.ndarray) -> np.ndarray:
        p, h, c = self.curve(beta)
        return -0.5 * np.einsum("ni,ij,nj->n", x, p, x) + x @ h + c

    @staticmethod
    def quadratic_moments(m, s, a, b, c):
        """E and Var of -x'Ax/2 + b'x + c for x ~ N(m, s)."""
        mean = -0.5 * (np.trace(a @ s) + m @ a @ m) + b @ m + c
        v = b - a @ m
        return mean, 0.5 * np.trace(a @ s @ a @ s) + v @ s @ v

    def score_moments(self, beta: float):
        """E and Var of d log f_beta / d beta under pi_beta."""
        m, s, _, _ = self.at(beta)
        return self.quadratic_moments(m, s, *self.slope(beta))

    def increment_moments(self, beta_from: float, beta_to: float):
        """E and Var of log f_to(x) - log f_from(x) for x ~ pi_from, one AIS weight increment."""
        m, s, _, _ = self.at(beta_from)
        p1, h1, c1 = self.curve(beta_from)
        p2, h2, c2 = self.curve(beta_to)
        return self.quadratic_moments(m, s, p2 - p1, h2 - h1, c2 - c1)

    def kl(self, beta1: float, beta2: float) -> float:
        m1, s1, _, _ = self.at(beta1)
        m2, s2, p2, _ = self.at(beta2)
        dm = m2 - m1
        return 0.5 * (np.trace(p2 @ s1) + dm @ p2 @ dm - self.d
                      + np.linalg.slogdet(s2)[1] - np.linalg.slogdet(s1)[1])


def endpoints(q_mean, q_cov, mean, cov):
    """(P, h, c) for a normalised q and for p~(x) = exp(-(x - mean)' cov^-1 (x - mean) / 2)."""
    p0 = np.linalg.inv(q_cov)
    c0 = -0.5 * q_mean @ p0 @ q_mean - 0.5 * len(mean) * LOG_2PI + 0.5 * np.linalg.slogdet(p0)[1]
    p1 = np.linalg.inv(cov)
    c1 = -0.5 * mean @ p1 @ mean
    return (p0, p0 @ q_mean, c0), (p1, p1 @ mean, c1)


def geometric(q_mean, q_cov, mean, cov) -> GaussianPath:
    """Natural parameters linear in beta: q^(1 - beta) p~^beta."""
    (p0, h0, c0), (p1, h1, c1) = endpoints(q_mean, q_cov, mean, cov)

    def curve(beta):
        return (1 - beta) * p0 + beta * p1, (1 - beta) * h0 + beta * h1, (1 - beta) * c0 + beta * c1

    def slope(_beta):
        return p1 - p0, h1 - h0, c1 - c0

    return GaussianPath("geometric", curve, slope, len(mean))


def moment_averaged(q_mean, q_cov, mean, cov) -> GaussianPath:
    """Mean and E[xx'] linear in beta (Grosse, Maddison and Salakhutdinov 2013).

    The covariance comes out as `(1 - beta) S0 + beta S1 + beta (1 - beta) dd'`,
    d the gap between the means, so the path widens along the mean shift halfway
    and is positive definite everywhere in [0, 1]. The constant is interpolated
    linearly, which changes nothing: it adds `c1 - c0` to every AIS run and every
    TI estimate whatever the schedule.
    """
    (_, _, c0), (_, _, c1) = endpoints(q_mean, q_cov, mean, cov)
    delta = mean - q_mean

    def cov_at(beta):
        return (1 - beta) * q_cov + beta * cov + beta * (1 - beta) * np.outer(delta, delta)

    def curve(beta):
        p = np.linalg.inv(cov_at(beta))
        return p, p @ (q_mean + beta * delta), (1 - beta) * c0 + beta * c1

    def slope(beta):
        p = np.linalg.inv(cov_at(beta))
        dp = -p @ (cov - q_cov + (1 - 2 * beta) * np.outer(delta, delta)) @ p
        return dp, dp @ (q_mean + beta * delta) + p @ delta, c1 - c0

    return GaussianPath("moment-avg", curve, slope, len(mean))


# ----------------------------------------------------------------------------
# Exact transitions, TI rules and schedules
# ----------------------------------------------------------------------------


def exact_bias(path: GaussianPath, betas: np.ndarray):
    """For exact transitions: log Z - E[log w] from the increments, the same from the KLs, and Var[log w]."""
    mean = var = kl_sum = 0.0
    for t in range(1, len(betas)):
        e, v = path.increment_moments(betas[t - 1], betas[t])
        mean += e
        var += v
        kl_sum += path.kl(betas[t - 1], betas[t])
    log_z = path.at(betas[-1])[3] - path.at(betas[0])[3]
    return log_z - mean, kl_sum, var


def profile(path: GaussianPath, points: int = 20001):
    grid = np.linspace(0.0, 1.0, points)
    moments = np.array([path.score_moments(beta) for beta in grid])
    return grid, moments[:, 0], moments[:, 1]


def simpson(values: np.ndarray, grid: np.ndarray) -> float:
    h = grid[1] - grid[0]
    return float(h / 3.0 * (values[0] + values[-1] + 4.0 * values[1:-1:2].sum() + 2.0 * values[2:-1:2].sum()))


def adaptive_schedule(grid: np.ndarray, var: np.ndarray, temps: int) -> np.ndarray:
    """Temperatures at equal steps of thermodynamic length, integral of sqrt(Var) d beta.

    To first order the exact-transition bias of a schedule is
    `sum (d beta)^2 Var / 2`, which for T steps is smallest when each step
    carries the same `d beta * sqrt(Var)`, and then it is `L^2 / 2T` against the
    uniform schedule's `J / 2T`. Cauchy-Schwarz gives `J >= L^2`, so `J / L^2`
    is everything a schedule can buy, and it is 1 only when Var is flat.
    """
    root = np.sqrt(var)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (root[1:] + root[:-1]) * np.diff(grid))])
    betas = np.interp(np.linspace(0.0, cum[-1], temps + 1), cum, grid)
    betas[0], betas[-1] = 0.0, 1.0
    return betas


def ti_trapezoid(path: GaussianPath, betas: np.ndarray):
    """Trapezoid TI on a schedule, and the sd it would have with one exact draw per temperature."""
    moments = np.array([path.score_moments(beta) for beta in betas])
    widths = np.zeros(len(betas))
    steps = np.diff(betas)
    widths[:-1] += 0.5 * steps
    widths[1:] += 0.5 * steps
    left = float(steps @ moments[:-1, 0])
    right = float(steps @ moments[1:, 0])
    return float(widths @ moments[:, 0]), left, right, float(np.sqrt(widths ** 2 @ moments[:, 1]))


def score_slope(path: GaussianPath, beta: float, eps: float = 1e-5) -> float:
    """d/d beta of E_beta[d log f / d beta], by central difference. For the geometric path it is Var."""
    return (path.score_moments(beta + eps)[0] - path.score_moments(beta - eps)[0]) / (2 * eps)


# ----------------------------------------------------------------------------
# AIS with MALA moves on any Gaussian path
# ----------------------------------------------------------------------------


def ais_mala(path: GaussianPath, betas: np.ndarray, runs: int, steps: int = 1, scale: float = 0.5):
    """Day 1's sampler on a general path. The weight increment is log f_t - log f_{t-1} at the current state."""
    d = path.d
    m0, s0, _, _ = path.at(betas[0])
    x = m0 + RNG.standard_normal((runs, d)) @ np.linalg.cholesky(s0).T
    log_w = np.zeros(runs)
    accepted = proposed = 0
    previous = path.log_f(betas[0], x)
    for t in range(1, len(betas)):
        current = path.log_f(betas[t], x)
        log_w += current - previous
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
        previous = path.log_f(betas[t], x)
    return log_w, accepted / max(proposed, 1)


# ----------------------------------------------------------------------------


def main():
    started = time.time()
    d = 8
    cov = ar1(d, 0.9)
    mean = np.linspace(-1.0, 1.0, d)
    prec = np.linalg.inv(cov)
    starts = {"cavi q": (mean, np.diag(1.0 / np.diag(prec))), "N(0, I)": (np.zeros(d), np.eye(d))}
    paths = {}
    for name, (m0, s0) in starts.items():
        paths[(name, "geometric")] = geometric(m0, s0, mean, cov)
        paths[(name, "moment-avg")] = moment_averaged(m0, s0, mean, cov)
    log_z = paths[("cavi q", "geometric")].at(1.0)[3]
    print(f"target: AR(1) r = 0.9, d = {d}   log Z = {log_z:.6f}")

    print("\n== checks ==")
    for name, (m0, s0) in starts.items():
        old = GeometricPath(m0, s0, mean, cov)
        new = paths[(name, "geometric")]
        worst = 0.0
        for beta in (0.0, 0.3, 0.9, 1.0):
            worst = max(worst, abs(old.at(beta)[3] - new.at(beta)[3]),
                        *np.abs(np.array(old.g_moments(beta)) - np.array(new.score_moments(beta))))
        assert worst < 1e-9
        for kind in ("geometric", "moment-avg"):
            path = paths[(name, kind)]
            assert abs(path.at(0.0)[3]) < 1e-10 and abs(path.at(1.0)[3] - log_z) < 1e-10
            m1, s1, _, _ = path.at(1.0)
            assert np.allclose(m1, mean) and np.allclose(s1, cov)
            by_mean, by_kl, _ = exact_bias(path, np.linspace(0.0, 1.0, 11))
            grid, e, _ = profile(path)
            integral = simpson(e, grid)
            print(f"  {name:8s} {kind:10s}  endpoints q and p   sum-KL identity {abs(by_mean - by_kl):.1e}"
                  f"   Simpson of E[d log f] on 20001 points - log Z {integral - log_z:+.1e}")
            assert abs(by_mean - by_kl) < 1e-10 and abs(integral - log_z) < 1e-8
        print(f"  {name:8s} general geometric path against day 1's closed forms, worst {worst:.1e}")

    print("\n== TI on a uniform schedule: AIS is the left Riemann sum ==")
    for name in starts:
        path = paths[(name, "geometric")]
        slope0, slope1 = score_slope(path, 0.0), score_slope(path, 1.0)
        _, var0 = path.score_moments(0.0)
        _, var1 = path.score_moments(1.0)
        assert abs(slope0 - var0) < 1e-4 * max(1.0, var0) and abs(slope1 - var1) < 1e-4 * max(1.0, var1)
        print(f"  {name}: Var at beta = 0 {var0:.4f}, at beta = 1 {var1:.4f}")
        print("       T    integral-left   right-integral   AIS exact bias   trapezoid err"
              "   (Var1-Var0)/12T^2   ratio   TI sd(1 draw/temp)   AIS sd(log w)")
        for T in (1, 2, 5, 10, 30, 100, 300, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            trap, left, right, ti_sd = ti_trapezoid(path, betas)
            ais_bias, _, ais_var = exact_bias(path, betas)
            day1_bias = exact_log_weight(GeometricPath(*starts[name], mean, cov), betas)[0]
            assert abs((log_z - left) - ais_bias) < 1e-10 and abs(ais_bias - day1_bias) < 1e-10
            assert abs((right - left) - (path.score_moments(1.0)[0] - path.score_moments(0.0)[0]) / T) < 1e-10
            predicted = (var1 - var0) / (12 * T * T)
            print(f"  {T:6d}   {log_z - left:12.6f}   {right - log_z:12.6f}   {ais_bias:12.6f}"
                  f"   {trap - log_z:+13.3e}   {predicted:+13.3e}   {(trap - log_z) / predicted:6.3f}"
                  f"   {ti_sd:14.4f}   {np.sqrt(ais_var):12.4f}")

    print("\n== the two paths, closed form ==")
    lengths = {}
    for (name, kind), path in paths.items():
        grid, e, var = profile(path)
        j = simpson(var, grid)
        # J on the geometric path is E_1[g] - E_0[g], which is KL(q || p) + KL(p || q). the
        # moment-averaged path reaches the same number from the dual side, asserted rather than assumed.
        jeffreys = path.kl(0.0, 1.0) + path.kl(1.0, 0.0)
        assert abs(j - jeffreys) < 1e-6
        length = float(np.trapezoid(np.sqrt(var), grid))
        lengths[(name, kind)] = (grid, var, j, length)
        low = float(np.trapezoid(var[grid <= 0.1], grid[grid <= 0.1])) / j
        high = float(np.trapezoid(var[grid >= 0.9], grid[grid >= 0.9])) / j
        peak = grid[np.argmax(var)]
        print(f"  {name:8s} {kind:10s}  J = {j:8.4f}   KL(q||p) + KL(p||q) = {jeffreys:8.4f}"
              f"   L = {length:7.4f}   L^2 = {length ** 2:8.4f}"
              f"   J/L^2 = {j / length ** 2:6.3f}   J in beta<0.1 {low:.3f}, beta>0.9 {high:.3f}"
              f"   Var peaks at beta {peak:.3f}, {var.max():.2f}")
    long_axis = np.linalg.eigh(cov)[1][:, -1]
    short_axis = np.linalg.eigh(cov)[1][:, 0]
    print("  variance of pi_beta along the target's longest and shortest axes")
    for beta in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        row = []
        for key in (("cavi q", "geometric"), ("cavi q", "moment-avg"), ("N(0, I)", "geometric"), ("N(0, I)", "moment-avg")):
            s = paths[key].at(beta)[1]
            row.append(f"{long_axis @ s @ long_axis:6.3f}/{short_axis @ s @ short_axis:5.3f}")
        print(f"    beta={beta:4.2f}   cavi geo {row[0]}   cavi MA {row[1]}   N(0,I) geo {row[2]}   N(0,I) MA {row[3]}")

    print("\n== uniform against adaptive schedules, exact transitions ==")
    schedules = {}
    for (name, kind), path in paths.items():
        grid, var, j, length = lengths[(name, kind)]
        print(f"  {name:8s} {kind:10s}  J/2 = {j / 2:8.4f}   L^2/2 = {length ** 2 / 2:8.4f}")
        for T in (10, 100, 1000):
            uniform = np.linspace(0.0, 1.0, T + 1)
            adaptive = adaptive_schedule(grid, var, T)
            schedules[(name, kind, T)] = adaptive
            b_uni = exact_bias(path, uniform)[1]
            b_ada = exact_bias(path, adaptive)[1]
            trap_ada = ti_trapezoid(path, adaptive)[0] - log_z
            trap_uni = ti_trapezoid(path, uniform)[0] - log_z
            share = np.mean(adaptive[1:] > 0.9)
            print(f"    T={T:5d}  uniform bias {b_uni:9.5f} (T*b {T * b_uni:8.4f})   adaptive {b_ada:9.5f}"
                  f" (T*b {T * b_ada:8.4f})   gain {b_uni / b_ada:6.3f}   temps past 0.9 {share:.3f}"
                  f"   TI trapezoid err uniform {trap_uni:+.2e}, adaptive {trap_ada:+.2e}")

    print("\n== with equal means the moment-averaged path is the geometric one run backwards ==")
    # whitened by q, the geometric precisions are I + beta (M - I) and the moment-averaged covariances
    # I + beta (M^-1 - I), with one set of eigenvectors, so per eigenvalue
    # KL_MA(a -> b) = KL_geo(1 - b -> 1 - a), and the sum of KLs along any schedule is the geometric
    # sum along its mirror image.
    geo, ma = paths[("cavi q", "geometric")], paths[("cavi q", "moment-avg")]
    other = np.random.default_rng(7)
    worst_step = worst_sum = worst_schedule = 0.0
    for _ in range(20):
        a, b = np.sort(other.random(2))
        worst_step = max(worst_step, abs(ma.kl(a, b) - geo.kl(1 - b, 1 - a)))
    for T in (10, 100, 1000):
        lopsided = np.concatenate([[0.0], np.sort(other.random(T - 1)), [1.0]])
        worst_sum = max(worst_sum, abs(exact_bias(ma, lopsided)[1] - exact_bias(geo, 1 - lopsided[::-1])[1]))
        mirrored = 1.0 - schedules[("cavi q", "geometric", T)][::-1]
        worst_schedule = max(worst_schedule, np.max(np.abs(schedules[("cavi q", "moment-avg", T)] - mirrored)))
    print(f"  one step, 20 random pairs, worst {worst_step:.1e}   sum along random schedules against their mirror,"
          f" worst {worst_sum:.1e}   adaptive MA schedule against 1 - reversed geometric, worst {worst_schedule:.1e}")
    assert worst_step < 1e-12 and worst_sum < 1e-10 and worst_schedule < 1e-6

    print("\n== MALA, one step per temperature ==")
    runs = 2000
    for (name, kind), path in paths.items():
        for T in (100, 1000):
            for label, betas in (("uniform", np.linspace(0.0, 1.0, T + 1)), ("adaptive", schedules[(name, kind, T)])):
                exact = exact_bias(path, betas)[1]
                log_w, acc = ais_mala(path, betas, runs)
                bias = log_z - log_w.mean()
                se = log_w.std(ddof=1) / np.sqrt(runs)
                print(f"  {name:8s} {kind:10s} T={T:5d} {label:8s}  bias {bias:7.4f} +- {se:.4f}"
                      f"   exact-transition {exact:7.4f}   ratio {bias / exact:6.2f}   accept {acc:.3f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
