"""Day 2 - adaptive tempering, each next beta chosen by bisection on a target ESS.

Day 1 ran SMC on a fixed uniform schedule. The usual way to run it is to let the
cloud pick the schedule, stepping each time to the largest beta whose increments
still have ESS rho N, and to resample at every step so that the ESS of the
increments is the conditional ESS. Same AR(1) target, same two starts, same
systematic resampling and MALA kernel as day 1, imported. The schedule is scored
against two fixed ones: the N = inf limit of the same rule, built from the closed
form r = E[w^2] / E[w]^2 of each increment, and AIS day 2's equal
thermodynamic-length schedule with the same T.

1. The number of temperatures is L / sqrt(log 1/rho) from N(0, I), and more
   than that from the CAVI q. If the increments were lognormal, ESS / N would be
   exp(-step^2 Var), so a fixed rho would be a fixed step of thermodynamic length.
   From N(0, I) the limit T is 5, 6, 8, 15, 49 and 156 for rho = 0.3 to 0.999
   against 4.5, 5.9, 8.3, 15.2, 49.3 and 156.4. From the CAVI q it is 7, 7, 9, 13,
   37 and 112 against 3.2, 4.2, 5.8, 10.7, 34.8 and 110.3. The CAVI q's beta* is
   1.019, so every increment near beta = 1 has an extrapolated temperature close
   to where r goes infinite, and the last step can never be longer than 0.019
   whatever rho asks for. That is why rho = 0.3 and 0.5 both come out at 7.
   At rho = 0.999 the limit schedule is within 0.004 and 0.007 in beta of the
   equal-length one, so at small steps the ESS rule is day 2's J / L^2 schedule
   found without knowing Var.

2. The adaptivity is most of the bias. At N = 100 and rho = 0.5 from the CAVI q
   the adaptive sampler is 0.402 low and the same rule's limit schedule, frozen
   and run on fresh clouds, is 0.022 low, 19x less. In Z itself the frozen
   schedule is unbiased, +0.002 +- 0.004, as it has to be, and the adaptive
   one is 0.29 low. It is 0.104 against 0.006 at rho = 0.9 and 0.078 against 0.003
   at N = 1000, so the gap shrinks by 5x for 10x the particles. From N(0, I) it is
   smaller, 0.089 against 0.026, and it is still the larger part. The frozen rows
   match day 1's first-order delta except at rho = 0.5 from the CAVI q, where r = 2
   per step is not small and the delta says 0.031 for a measured 0.022.

   What the sample does differently is take fewer steps, 4.2 at N = 100 against
   the limit's 7, and 5.7 at N = 1000. A cloud that has not drawn the tail of w
   reports a higher ESS than the population has, so it steps further, and the
   increment it then estimates is from the same cloud that missed the tail. I
   have not measured the sample ESS against the population one directly. The
   step past beta*, which the sample ESS cannot see, happens less often than I
   expected: 10 of 20 runs at rho = 0.3 and 4 of 20 at rho = 0.5 from the CAVI q,
   18 of 20 at rho = 0.3 from N(0, I), none at rho = 0.7 or above.

3. With MALA the frozen limit schedule beats the adaptive one from the CAVI q.
   At rho = 0.9 and 10 steps per temperature the adaptive sampler is 0.288 low on
   10.9 temperatures and the limit schedule 0.019 low on 13, three standard
   errors apart. The equal-length schedule is 0.023 low on 11 with an sd of 0.64,
   3x the others. The adaptive cloud ends with a long-axis variance of 3.25 of
   6.20 against 4.23 for the frozen limit, so the adaptive rule is choosing its
   steps on a cloud that is narrower than pi_beta and stepping too far for the
   same reason as in 2. At rho = 0.99 the four schedules are inside their noise.
   From N(0, I) no row separates from any other at any rho.

Six predictions written before the run. None right, two half, four wrong.

- Half: sample T within 10% of L / sqrt(log 1/rho) for rho >= 0.9. It is within
  2% from N(0, I) and within 6% at rho = 0.99 from the CAVI q, and 17% over at
  rho = 0.9 from the CAVI q.
- Half: the limit schedule within 0.02 in beta of the equal-length one at
  rho = 0.99. 0.0165 from the CAVI q, 0.0207 from N(0, I).
- Wrong: over half the CAVI q runs at rho = 0.5 take a last step past beta*. 4 of 20.
- Wrong: adaptive and frozen log Z bias not separable at 2 se at N = 100. They
  are 59 se apart from the CAVI q and 12 from N(0, I).
- Wrong: MALA picks at least 20% fewer temperatures than exact moves from the
  CAVI q. 20.0% at rho = 0.5, 13% at 0.9 and 2% at 0.99.
- Wrong: MALA adaptive no more biased than the frozen limit schedule. It is 0.27
  more biased at rho = 0.9 from the CAVI q, three standard errors.

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
from day2_thermodynamic_integration import adaptive_schedule  # noqa: E402
import day1_tempered_smc as day1  # noqa: E402

RNG = np.random.default_rng(22)
day1.RNG = RNG


def log_sum_exp(a: np.ndarray) -> float:
    top = a.max()
    return float(top + np.log(np.sum(np.exp(a - top))))


# ----------------------------------------------------------------------------
# Choosing the next temperature
# ----------------------------------------------------------------------------


def ess_fraction(g: np.ndarray, step: float) -> float:
    """ESS / N of the increments exp(step * g) over an equally weighted cloud."""
    a = step * g
    return float(np.exp(2.0 * log_sum_exp(a) - log_sum_exp(2.0 * a) - np.log(len(g))))


def next_beta_sample(g: np.ndarray, beta: float, rho: float, iters: int = 60) -> float:
    """The largest next beta whose sample ESS fraction is still rho, by bisection.

    The cloud is resampled after every step, so it comes in equally weighted and
    the ESS of the incremental weights is the conditional ESS. ESS / N falls
    monotonically in the step here because every increment is the same g scaled,
    and a larger scale only spreads the weights.
    """
    if ess_fraction(g, 1.0 - beta) >= rho:
        return 1.0
    lo, hi = 0.0, 1.0 - beta
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ess_fraction(g, mid) >= rho:
            lo = mid
        else:
            hi = mid
    return beta + lo


def log_r(path: GeometricPath, a: float, b: float) -> float:
    """log E[w^2] / E[w]^2 for the increment from a to b under exact pi_a, inf past beta*."""
    ext = path.at(2.0 * b - a)
    if ext is None:
        return np.inf
    return ext[3] + path.at(a)[3] - 2.0 * path.at(b)[3]


def limit_schedule(path: GeometricPath, rho: float, iters: int = 60) -> np.ndarray:
    """The schedule the adaptive rule converges to as N grows, with exact moves.

    The population ESS fraction of one increment is E[w]^2 / E[w^2] = 1 / r, so
    the N = inf rule steps until log r = log(1 / rho). A step whose
    extrapolated temperature is past beta* has r = inf and is always too long,
    which the sample ESS of a finite cloud never sees.
    """
    target = np.log(1.0 / rho)
    betas = [0.0]
    while betas[-1] < 1.0:
        a = betas[-1]
        if log_r(path, a, 1.0) <= target:
            betas.append(1.0)
            break
        lo, hi = 0.0, 1.0 - a
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if log_r(path, a, a + mid) <= target:
                lo = mid
            else:
                hi = mid
        betas.append(a + lo)
    return np.array(betas)


# ----------------------------------------------------------------------------
# The sampler
# ----------------------------------------------------------------------------


def draw(path: GeometricPath, beta: float, n: int) -> np.ndarray:
    m, s, _, _ = path.at(beta)
    return m + RNG.standard_normal((n, path.d)) @ np.linalg.cholesky(s).T


def smc(path: GeometricPath, n: int, move, rho=None, betas=None):
    """Reweight, resample, move, with the schedule either fixed or chosen on the fly.

    Pass `betas` for a fixed schedule and `rho` for the adaptive one. `move` is
    None for exact draws, else a number of MALA steps with day 1's kernel. The
    cloud is resampled at every step, so log Z_hat is a sum of `log mean exp`
    of the increments. Returns log Z_hat, the schedule used, the final cloud and
    the number of distinct starting ancestors.
    """
    x = draw(path, 0.0, n)
    ancestor = np.arange(n)
    used = [0.0]
    log_z = 0.0
    t = 0
    while used[-1] < 1.0:
        t += 1
        g = path.g(x)
        beta = next_beta_sample(g, used[-1], rho) if betas is None else betas[t]
        inc = (beta - used[-1]) * g
        log_z += log_sum_exp(inc) - np.log(n)
        w = np.exp(inc - inc.max())
        idx = day1.systematic(w / w.sum())
        x, ancestor = x[idx], ancestor[idx]
        used.append(beta)
        if beta >= 1.0:
            break
        if move is None:
            x = draw(path, beta, n)
        else:
            x, _ = day1.mala(path, x, beta, move)
    return log_z, np.array(used), x, len(np.unique(ancestor))


def summarise(rows, path, axis):
    """Mean and sd of log Z_hat - log Z, mean of Z_hat / Z - 1, mean T, long-axis variance, ancestors."""
    err = np.array([r[0] for r in rows]) - path.log_z()
    temps = np.array([len(r[1]) - 1 for r in rows])
    var_long = np.mean([np.var(r[2] @ axis) for r in rows])
    return err, temps, var_long, np.mean([r[3] for r in rows])


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
    rhos = (0.3, 0.5, 0.7, 0.9, 0.99, 0.999)
    grids = {}
    print(f"target: AR(1) r = 0.9, d = {d}   log Z = {starts['cavi q'].log_z():.6f}")

    print("\n== how many temperatures, exact moves ==")
    print("  lognormal increments give ESS / N = exp(-step^2 Var), so a fixed rho is a fixed step of"
          " thermodynamic length and T ~ L / sqrt(log 1/rho)")
    for name, path in starts.items():
        grid = np.linspace(0.0, 1.0, 20001)
        var = np.array([path.g_moments(b)[1] for b in grid])
        length = float(np.trapezoid(np.sqrt(var), grid))
        j = float(np.trapezoid(var, grid))
        grids[name] = (grid, var)
        print(f"  {name}: L = {length:.4f}   J = {j:.4f}   J/L^2 = {j / length ** 2:.3f}"
              f"   beta* = {path.beta_star():.4f}")
        print("      rho   L/sqrt(log 1/rho)   limit T   sample T (N=1000, 20 runs)   limit vs equal-length"
              " schedule at the same T, worst |d beta|   runs whose last step is past beta*")
        for rho in rhos:
            lim = limit_schedule(path, rho)
            eq = adaptive_schedule(grid, var, len(lim) - 1)
            temps, past = [], 0
            for _ in range(20):
                _, used, _, _ = smc(path, 1000, None, rho=rho)
                temps.append(len(used) - 1)
                past += int(not np.isfinite(log_r(path, used[-2], used[-1])))
            print(f"    {rho:5.3f}   {length / np.sqrt(np.log(1 / rho)):10.2f}   {len(lim) - 1:8d}"
                  f"   {np.mean(temps):7.2f} (min {min(temps)}, max {max(temps)})"
                  f"   {np.max(np.abs(lim - eq)):.4f}   {past:3d}/20")

    print("\n== what the adaptivity costs, exact moves: adaptive against its own limit schedule frozen ==")
    print("  start     rho     N   sampler    log Z_hat - log Z        sd    Z_hat/Z - 1         mean T")
    for name, path in starts.items():
        for rho, n, reps in ((0.5, 100, 4000), (0.9, 100, 4000), (0.5, 1000, 1000)):
            lim = limit_schedule(path, rho)
            for label in ("adaptive", "frozen"):
                if label == "adaptive":
                    rows = [smc(path, n, None, rho=rho) for _ in range(reps)]
                else:
                    rows = [smc(path, n, None, betas=lim) for _ in range(reps)]
                err, temps, _, _ = summarise(rows, path, axis)
                ratio = np.exp(err) - 1.0
                print(f"  {name:8s} {rho:4.2f} {n:5d}   {label:8s}   {err.mean():+.4f} +- {err.std(ddof=1) / np.sqrt(reps):.4f}"
                      f"   {err.std(ddof=1):.4f}   {ratio.mean():+.4f} +- {ratio.std(ddof=1) / np.sqrt(reps):.4f}"
                      f"   {temps.mean():6.2f}")
            print(f"  {'':8s} {rho:4.2f} {n:5d}   first-order delta for the frozen schedule"
                  f" {-day1.delta_bias(day1.step_second_moments(path, lim), n, True):+.4f}")

    print("\n== MALA, 10 steps per temperature, N = 1000, 10 runs per row ==")
    print("  start     rho   schedule      T     log Z_hat - log Z     sd    long-axis var  ancestors")
    for name, path in starts.items():
        for rho in (0.5, 0.9, 0.99):
            rows = [smc(path, 1000, 10, rho=rho) for _ in range(10)]
            err, temps, v, a = summarise(rows, path, axis)
            t_ad = int(round(temps.mean()))
            print(f"  {name:8s} {rho:4.2f}   adaptive   {temps.mean():6.1f}   {err.mean():+.4f} +- {err.std(ddof=1) / np.sqrt(10):.4f}"
                  f"   {err.std(ddof=1):.4f}   {v:8.3f}   {a:7.1f}")
            grid, var = grids[name]
            others = (("limit", limit_schedule(path, rho)),
                      ("equal-L", adaptive_schedule(grid, var, t_ad)),
                      ("uniform", np.linspace(0.0, 1.0, t_ad + 1)))
            for label, betas in others:
                rows = [smc(path, 1000, 10, betas=betas) for _ in range(10)]
                err, temps, v, a = summarise(rows, path, axis)
                print(f"  {'':8s} {'':4s}   {label:8s}   {temps.mean():6.1f}   {err.mean():+.4f} +- {err.std(ddof=1) / np.sqrt(10):.4f}"
                      f"   {err.std(ddof=1):.4f}   {v:8.3f}   {a:7.1f}")
    print(f"  p's long-axis variance {axis @ cov @ axis:.3f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
