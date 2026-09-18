"""Day 3 - bidirectional Monte Carlo, and whether the sandwich reports the gap.

Forward AIS from q is biased low in log Z. Run the same path backwards from an
exact draw of p and the log weight estimates `-log Z`, biased low as well, so
minus it is an upper bound in expectation (Grosse, Ghahramani and Adams 2015).
The two together are a sandwich, and the question from the project description
is whether its width reports the gap the ELBO could not. Here exact draws of p
exist, which is the thing a real model does not have, so the last section
starts the reverse run from states that are only close to p. Same AR(1) target,
same two starts, day 2's paths.

1. With exact transitions the upper side is the sum of the KLs the other way,
   `KL(pi_t || pi_{t-1})`, asserted against the increment moments at every T. At
   T = 1 it is the EUBO gap `KL(p || q)`, so the width at T = 1 is J. On a
   uniform geometric schedule the width is J / T exactly at every T, asserted to
   1e-9, since it is day 2's right Riemann sum minus its left one. The simulated
   sides sit inside 4 standard errors of the closed forms on all twelve rows.

2. The width does not split evenly until T is large. From the CAVI start the
   upper side is 10.4x the lower at T = 1, 2.56x at T = 10 and 1.17x at T = 100,
   because the CAVI q is the one that is close in KL(q || p) and far in
   KL(p || q). From N(0, I) it is the other way, 0.29x at T = 1. So at T = 1 the
   width reports the ELBO's gap as 11.4x too large from the CAVI start and 1.29x
   from N(0, I), and from T = 300 it is 2.0x from both. The adaptive schedule
   shrinks the width by the same J / L^2 day 2 found for the lower side, 2.454
   and 1.369 from T = 100, and the moment-averaged path gives the same width as the geometric
   one on a uniform schedule from both starts, 0.29842 and 0.33474 at T = 100.

3. Under MALA the sandwich is still a sandwich, and far wider than the closed
   form. From the CAVI start at T = 1000 the width is 1.797 against 0.030
   exact, 0.517 below and 1.280 above, and the reverse run is 85x its
   exact-transition bias where the forward run is 35x. From N(0, I) it is
   0.680, 0.301 below and 0.379 above. So on the case the ELBO misjudged most,
   the width over-reports the lower side's bias 3.5x, where from N(0, I) it is
   2.3x. It never under-reports it, which is what a bound is for.

4. That needs exact draws of p. Start the reverse run from the forward MALA
   run's final states and the upper side stops being one. From the CAVI start
   at T = 100 it is 0.230 +- 0.037 below log Z, so both sides of the sandwich are
   below the truth and the interval excludes it. From draws of q it is 1.07
   below. The forward finals have long-axis variance 0.63 against the target's
   6.20, and at T = 1000, where the forward chains have reached 1.75, the upper
   side is 0.071 +- 0.028 above log Z and the sandwich is 0.59 wide against the
   1.80 an exact start gives. It contains log Z and reads 3x more certain than
   it is. From N(0, I), whose chains start wider on that axis, the same
   substitution stays above log Z at T = 100 and 1000.

Five predictions written before the run. Four right, one half.

- Right: width = J / T on a uniform geometric schedule. From the CAVI start the
  upper side is larger than the lower at every T up to 100. Under MALA at
  T = 1000 the width is more than 2x the lower side's bias from both starts,
  3.47 and 2.26.
- Right: started from the forward MALA finals at T = 100 from the CAVI start,
  the upper side falls below log Z.
- Half: the reverse run's ratio to its exact-transition bias within 2x of the
  forward run's. From N(0, I) it held at 0.87x, 1.22x and 1.26x. From the CAVI
  start it is 3.4x, 4.2x and 2.4x at T = 10, 100 and 1000. The reverse run from
  the CAVI start crosses beta > 0.9, where 79% of J sits, in its first tenth of
  steps, starting from p at long-axis variance 6.2 and having to shrink to 1.0.
  That is where it spends its bias, but I have not checked that this is why.

NumPy only. Fixed seeds. About 35 seconds.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day1_ais_gaussians import ar1  # noqa: E402
from day2_thermodynamic_integration import GaussianPath, exact_bias, geometric, moment_averaged  # noqa: E402

RNG = np.random.default_rng(3)


# ----------------------------------------------------------------------------
# Both directions with exact transitions, closed form
# ----------------------------------------------------------------------------


def exact_reverse(path: GaussianPath, betas: np.ndarray):
    """Reverse AIS from an exact draw of pi_1, exact transitions: U - log Z two ways, and Var of log w_rev.

    The reverse run visits the temperatures from 1 down to 0 and at each one adds
    `log f_{t-1}(x) - log f_t(x)` with x an exact draw of pi_t, so its log weight
    estimates `log Z_0 - log Z_1 = -log Z`. U is minus its mean. Each increment has
    expectation `log Z_{t-1} - log Z_t + KL(pi_t || pi_{t-1})`, so U sits above
    log Z by the sum of the KLs in the other direction from the forward run's.
    """
    mean = var = kl_sum = 0.0
    for t in range(1, len(betas)):
        e, v = path.increment_moments(betas[t], betas[t - 1])
        mean += e
        var += v
        kl_sum += path.kl(betas[t], betas[t - 1])
    log_z = path.at(betas[-1])[3] - path.at(betas[0])[3]
    return -mean - log_z, kl_sum, var


def sandwich(path: GaussianPath, betas: np.ndarray):
    """(lower bias, upper bias, width) with exact transitions. Width = sum of symmetric KLs between neighbours."""
    lower = exact_bias(path, betas)[1]
    upper = exact_reverse(path, betas)[1]
    return lower, upper, lower + upper


# ----------------------------------------------------------------------------
# Either direction under MALA
# ----------------------------------------------------------------------------


def anneal(path: GaussianPath, betas: np.ndarray, x: np.ndarray, steps: int = 1, scale: float = 0.5):
    """Run AIS along `betas` in the order given, from states x, with MALA at every interior temperature.

    Forward is betas increasing from q's draws and the log weight estimates
    log Z; reverse is betas decreasing from p's draws and it estimates -log Z.
    One function, so that the two directions differ only in where they start
    and which way they walk. Returns the log weights and the final states.
    """
    x = x.copy()
    runs = len(x)
    log_w = np.zeros(runs)
    previous = path.log_f(betas[0], x)
    for t in range(1, len(betas)):
        log_w += path.log_f(betas[t], x) - previous
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
        previous = path.log_f(betas[t], x)
    return log_w, x


def draw(path: GaussianPath, beta: float, n: int) -> np.ndarray:
    m, s, _, _ = path.at(beta)
    return m + RNG.standard_normal((n, path.d)) @ np.linalg.cholesky(s).T


# ----------------------------------------------------------------------------


def main():
    started = time.time()
    d = 8
    cov = ar1(d, 0.9)
    mean = np.linspace(-1.0, 1.0, d)
    prec = np.linalg.inv(cov)
    starts = {"cavi q": (mean, np.diag(1.0 / np.diag(prec))), "N(0, I)": (np.zeros(d), np.eye(d))}
    geo = {name: geometric(m0, s0, mean, cov) for name, (m0, s0) in starts.items()}
    ma = {name: moment_averaged(m0, s0, mean, cov) for name, (m0, s0) in starts.items()}
    log_z = geo["cavi q"].at(1.0)[3]
    print(f"target: AR(1) r = 0.9, d = {d}   log Z = {log_z:.6f}")

    print("\n== exact transitions: the two sides and the width ==")
    for name, path in geo.items():
        e0 = path.score_moments(0.0)[0]
        e1 = path.score_moments(1.0)[0]
        j = e1 - e0
        kl_qp, kl_pq = path.kl(0.0, 1.0), path.kl(1.0, 0.0)
        print(f"  {name}: KL(q||p) {kl_qp:.4f}   KL(p||q) {kl_pq:.4f}   J {j:.4f}   EUBO - log Z at T=1 is KL(p||q)")
        print("        T    lower bias     upper bias      width     T*width    upper/lower   width/lower"
              "   sd(log w) fwd   sd rev")
        for T in (1, 2, 5, 10, 30, 100, 300, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            by_mean, by_kl, var_rev = exact_reverse(path, betas)
            assert abs(by_mean - by_kl) < 1e-9
            lower, upper, width = sandwich(path, betas)
            if T == 1:
                assert abs(lower - kl_qp) < 1e-10 and abs(upper - kl_pq) < 1e-10
            # on the geometric path the width is right sum minus left sum of E_beta[g], so J / T on a uniform schedule.
            assert abs(width - j / T) < 1e-9
            var_fwd = exact_bias(path, betas)[2]
            print(f"  {T:7d}   {lower:11.6f}   {upper:11.6f}   {width:10.6f}   {T * width:8.4f}"
                  f"   {upper / lower:10.3f}   {width / lower:10.3f}   {np.sqrt(var_fwd):12.4f}   {np.sqrt(var_rev):8.4f}")

    print("\n== the width on a non-uniform schedule and on the other path ==")
    for name in starts:
        grid = np.linspace(0.0, 1.0, 20001)
        var = np.array([geo[name].score_moments(beta)[1] for beta in grid])
        root = np.sqrt(var)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (root[1:] + root[:-1]) * np.diff(grid))])
        for T in (10, 100, 1000):
            uniform = np.linspace(0.0, 1.0, T + 1)
            adaptive = np.interp(np.linspace(0.0, cum[-1], T + 1), cum, grid)
            adaptive[0], adaptive[-1] = 0.0, 1.0
            lo_u, up_u, w_u = sandwich(geo[name], uniform)
            lo_a, up_a, w_a = sandwich(geo[name], adaptive)
            lo_m, up_m, w_m = sandwich(ma[name], uniform)
            print(f"  {name:8s} T={T:5d}  geometric uniform {lo_u:.5f} + {up_u:.5f} = {w_u:.5f}"
                  f"   adaptive {lo_a:.5f} + {up_a:.5f} = {w_a:.5f} (width gain {w_u / w_a:5.3f})"
                  f"   moment-avg uniform {lo_m:.5f} + {up_m:.5f} = {w_m:.5f}")

    print("\n== exact transitions, simulated ==")
    for name, path in geo.items():
        for T, runs in ((1, 20000), (10, 20000), (100, 4000)):
            betas = np.linspace(0.0, 1.0, T + 1)
            lower, upper, _ = sandwich(path, betas)
            # exact transitions: each state is a fresh draw at its temperature, independent of the last
            fwd = sum(path.log_f(betas[t], x) - path.log_f(betas[t - 1], x)
                      for t in range(1, T + 1) for x in [draw(path, betas[t - 1], runs)])
            rev = sum(path.log_f(betas[t - 1], x) - path.log_f(betas[t], x)
                      for t in range(1, T + 1) for x in [draw(path, betas[t], runs)])
            se_f = fwd.std(ddof=1) / np.sqrt(runs)
            se_r = rev.std(ddof=1) / np.sqrt(runs)
            print(f"  {name:8s} T={T:4d}  log Z - E log w {log_z - fwd.mean():8.4f} +- {se_f:.4f} (exact {lower:.4f})"
                  f"   U - log Z {-rev.mean() - log_z:8.4f} +- {se_r:.4f} (exact {upper:.4f})")
            assert abs(log_z - fwd.mean() - lower) < 4 * se_f and abs(-rev.mean() - log_z - upper) < 4 * se_r

    print("\n== MALA, one step per temperature: forward from q, reverse from exact p ==")
    runs = 2000
    finals = {}
    for name, path in geo.items():
        for T in (10, 100, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            lower, upper, width = sandwich(path, betas)
            fwd, x_end = anneal(path, betas, draw(path, 0.0, runs))
            rev, _ = anneal(path, betas[::-1], draw(path, 1.0, runs))
            finals[(name, T)] = x_end
            lo = log_z - fwd.mean()
            up = -rev.mean() - log_z
            se_lo = fwd.std(ddof=1) / np.sqrt(runs)
            se_up = rev.std(ddof=1) / np.sqrt(runs)
            print(f"  {name:8s} T={T:5d}  lower {lo:7.4f} +- {se_lo:.4f} ({lo / lower:5.2f}x exact)"
                  f"   upper {up:7.4f} +- {se_up:.4f} ({up / upper:5.2f}x exact)"
                  f"   width {lo + up:7.4f} (exact {width:.4f})   width/lower {(lo + up) / lo:5.2f}")

    print("\n== reverse AIS from states that are not exact draws of p ==")
    long_axis = np.linalg.eigh(cov)[1][:, -1]
    for name, path in geo.items():
        for T in (10, 100, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            upper = sandwich(path, betas)[1]
            sources = {"forward MALA T=100 finals": finals[(name, 100)],
                       f"forward MALA T={T} finals": finals[(name, T)],
                       "draws of q": draw(path, 0.0, runs)}
            for label, x0 in sources.items():
                rev, _ = anneal(path, betas[::-1], x0)
                up = -rev.mean() - log_z
                se = rev.std(ddof=1) / np.sqrt(runs)
                print(f"  {name:8s} T={T:5d} from {label:26s}  U - log Z {up:+8.4f} +- {se:.4f}"
                      f"   (exact-start, exact-transition {upper:.4f})   start long-axis var {np.var(x0 @ long_axis):6.3f}"
                      f" of {long_axis @ cov @ long_axis:.3f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
