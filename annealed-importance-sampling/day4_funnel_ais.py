"""Day 4 - Neal's funnel, started from the VI project's day 4 q.

VI day 4 left a Gaussian q on the funnel (sigma_v = 3, d = 3) whose gap is
closed form, `KL(q || p) = 0.5 log(1 + k sigma_v^2 / 2) = 1.151293`, and whose
own importance estimate of log Z came back 0.33 low, 0.46 after PSIS. The funnel
is normalised, so log Z = 0 and every AIS estimate is its own error. Nothing on
the geometric path is Gaussian any more, so there are no exact-transition
numbers to score against; the transitions are HMC, the MCMC project's sampler,
with its divergence count kept at every temperature. p itself can be drawn
exactly, `v ~ N(0, 9)` and then x given v, so the reverse run from day 3 still
has its exact start.

1. The two sides at T = 1 are 120x apart. KL(q || p) is 1.1518 by 200000
   draws against the closed form 1.151293, and KL(p || q) is 138.4, because
   the reverse-KL q has Var(x) 0.638 where p has 90. The moment-matched q is the
   mirror image, 8062 and 4.35. So J is 139.6 from one and 8066 from the other,
   and on a uniform path the reverse-KL q is the only start AIS can use.

2. The forward bias does not fall like 1/T. From the reverse-KL q with HMC at
   eps 0.2 it is 0.446, 0.241, 0.135 at T = 10, 100, 1000, which is 1.8x per
   decade where a sampler that kept up would give 10x. The runs do not reach the
   neck: p has 8.3% of its mass below v = -4.16 and the final states have 0.6%,
   2.2% and 4.75%, min v -5.43 against p's 2.3% below -6. And `log mean w`
   over the 2000 runs does not move with T at all, -0.076, -0.112, -0.086. So
   against VI day 4's plain IS, 0.330 low, the path out of q takes the error to
   about 0.09 and then stops, which is the missing neck mass a mean of weights
   cannot put back. No standard error on that plateau; the three numbers are
   single runs of 2000.

3. The step size trades one failure for another. eps 0.05 reaches min v -7.37
   with 11 divergences and is further off, 0.208, because it moves less per
   temperature. eps 0.4 has 5926 divergences, min v -4.43 and 0.160. eps 0.1 and
   0.2 are the same, 0.136 and 0.135. None of the four closes the gap below 0.13.

4. The divergences sit where the path opens the neck, from the reverse-KL q:
   709 of 892 at beta > 0.9 at T = 1000. From the moment-matched q it is the
   other way round, 15039 of 17528 at beta <= 0.9, because that q starts with
   Var(x) 90 at v near -9 and the first temperatures are already in the neck.

5. The sandwich still holds and still says little. Reverse AIS from exact draws
   of p puts the upper side at 13.9, 6.6 and 2.22 above log Z at T = 10, 100,
   1000, so the width is 18.6x to 32x the lower side's actual error, and the
   reverse runs diverge 38100 times at T = 1000, 19 per run. Started from the
   forward finals instead, the upper side is -0.092 and -0.045 at T = 100 and
   1000, both sides below log Z and an interval 0.09 wide that excludes it,
   which is day 3's section 4 again on a target where it matters.

6. The moment-matched q is the ranking problem from VI day 4 in another form.
   Its mean log weight is thousands of nats off with a standard error in the
   thousands, sd 67949 at T = 100, so the lower side of the sandwich is useless.
   Its `log mean w` is -0.207, -0.008 and +0.260, the closest of any run at
   T = 100 and above log Z at T = 1000. The ELBO, the AIS lower bound and the
   log of the mean weight put the two starts in different orders.

Five predictions written before the run. One right, one half, three wrong.

- Right: the width at T = 100 from the reverse-KL q more than 3x the lower
  side's error. It is 29x.
- Half: most divergences at beta > 0.9. From the reverse-KL q, 79% of them.
  From the moment-matched q, 14%.
- Wrong: T = 1000 closes the gap to under 0.1. It is 0.135, and no step size
  gets below 0.136.
- Wrong: the final states hold less than half of p's mass below v = -4.16. They
  hold 4.75% of 8.26%, just over half.
- Wrong: the moment-matched q ahead at T = 10 and behind at T = 1000. On
  E log w it is behind everywhere by thousands of nats, and on log mean w it is
  behind at T = 10 and ahead at 100.

NumPy only. Fixed seeds. About 40 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "variational-inference-from-scratch"))
from day4_funnel import funnel_grad, funnel_log_p, gaussian_log_q, mean_field_optimum, moment_matched  # noqa: E402

RNG = np.random.default_rng(4)
SIGMA_V = 3.0
D = 3


class FunnelPath:
    """f_beta = q^(1 - beta) p^beta for a Gaussian q and the normalised funnel.

    Both ends integrate to one, so log Z_0 = log Z_1 = 0 and the forward log
    weight is an estimate of 0 whose mean is minus the bias. Only the density
    and its gradient are needed, which is all a real model would give.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean, self.cov = mean, cov
        self.prec = np.linalg.inv(cov)
        self.chol = np.linalg.cholesky(cov)

    def log_q(self, z):
        return gaussian_log_q(z, self.mean, self.cov)

    def log_f(self, beta, z):
        return (1.0 - beta) * self.log_q(z) + beta * funnel_log_p(z, SIGMA_V)

    def grad(self, beta, z):
        return -(1.0 - beta) * (z - self.mean) @ self.prec + beta * funnel_grad(z, SIGMA_V)

    def draw_q(self, n):
        return self.mean + RNG.standard_normal((n, D)) @ self.chol.T


def draw_p(n):
    """Exact draws of the funnel, v in the last column."""
    v = SIGMA_V * RNG.standard_normal(n)
    x = RNG.standard_normal((n, D - 1)) * np.exp(0.5 * v)[:, None]
    return np.column_stack([x, v])


def hmc_step(path: FunnelPath, beta, z, eps, n_leap, threshold=1000.0):
    """One HMC move at temperature beta for every run at once. Returns (z, accepted, diverged).

    A divergence is an energy error above MCMC day 4's threshold or not finite,
    the same definition, so the counts can be read against that project's.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        p0 = RNG.standard_normal(z.shape)
        h0 = -path.log_f(beta, z) + 0.5 * np.sum(p0 * p0, axis=1)
        q, p = z.copy(), p0 + 0.5 * eps * path.grad(beta, z)
        for i in range(n_leap):
            q = q + eps * p
            g = path.grad(beta, q)
            p = p + (eps if i < n_leap - 1 else 0.5 * eps) * g
        h1 = -path.log_f(beta, q) + 0.5 * np.sum(p * p, axis=1)
        dh = h1 - h0
    bad = ~np.isfinite(dh)
    diverged = bad | (np.nan_to_num(dh, nan=0.0) > threshold)
    dh = np.where(bad, np.inf, dh)
    accept = np.log(RNG.random(len(z))) < -dh
    z = np.where(accept[:, None], q, z)
    return z, accept, diverged


def anneal(path: FunnelPath, betas, z, eps=0.2, n_leap=10):
    """AIS along betas in the order given, one HMC move per interior temperature.

    Forward from q estimates log Z = 0, reverse from p estimates -log Z = 0.
    Returns log w, final states, acceptance rate, and the divergences split into
    beta <= 0.9 and beta > 0.9.
    """
    z = z.copy()
    log_w = np.zeros(len(z))
    previous = path.log_f(betas[0], z)
    acc = n = div_low = div_high = 0
    for t in range(1, len(betas)):
        log_w += path.log_f(betas[t], z) - previous
        if t == len(betas) - 1:
            break
        z, a, dv = hmc_step(path, betas[t], z, eps, n_leap)
        acc += int(a.sum())
        n += len(z)
        if betas[t] > 0.9:
            div_high += int(dv.sum())
        else:
            div_low += int(dv.sum())
        previous = path.log_f(betas[t], z)
    return log_w, z, acc / max(n, 1), div_low, div_high


def log_mean_exp(a):
    top = a.max()
    return top + np.log(np.mean(np.exp(a - top)))


def main():
    started = time.time()
    k = D - 1
    qs = {"reverse-KL q": mean_field_optimum(SIGMA_V, D), "moment-matched q": moment_matched(SIGMA_V, D)}
    paths = {name: FunnelPath(m, c) for name, (m, c) in qs.items()}
    gap = 0.5 * np.log(1.0 + k * SIGMA_V**2 / 2.0)
    print(f"funnel sigma_v = {SIGMA_V}, d = {D}, log Z = 0   reverse-KL q's closed-form gap {gap:.6f}")

    # p's mass below where q's draws stop, for reading the final states against
    ref = draw_p(400000)[:, -1]
    print(f"  p: P(v < -4.16) {np.mean(ref < -4.16):.4f}   P(v < -6) {np.mean(ref < -6):.4f}")

    print("\n== T = 1 is the ELBO, and KL(p || q) is the other side ==")
    for name, path in paths.items():
        zq = path.draw_q(200000)
        zp = draw_p(200000)
        lower = -np.mean(funnel_log_p(zq, SIGMA_V) - path.log_q(zq))
        upper = np.mean(funnel_log_p(zp, SIGMA_V) - path.log_q(zp))
        print(f"  {name:17s} KL(q||p) {lower:9.4f}   KL(p||q) {upper:9.4f}   J = {lower + upper:9.4f}")
        if name == "reverse-KL q":
            assert abs(lower - gap) < 0.01

    runs = 2000
    print(f"\n== forward AIS with HMC, eps 0.2, 10 leapfrog steps, {runs} runs ==")
    finals = {}
    for name, path in paths.items():
        for T in (10, 100, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            log_w, z, acc, dl, dh = anneal(path, betas, path.draw_q(runs))
            finals[(name, T)] = z
            se = log_w.std(ddof=1) / np.sqrt(runs)
            v = z[:, -1]
            print(f"  {name:17s} T={T:5d}  log Z - E log w {-log_w.mean():7.4f} +- {se:.4f}"
                  f"   log mean w {log_mean_exp(log_w):+7.4f}   sd {log_w.std(ddof=1):6.3f}   accept {acc:.3f}"
                  f"   div beta<=0.9 {dl:6d}  >0.9 {dh:6d}   final Var(v) {v.var():5.2f}"
                  f"   P(v<-4.16) {np.mean(v < -4.16):.4f}   min v {v.min():6.2f}")

    print("\n== the step size, reverse-KL q, T = 1000 ==")
    path = paths["reverse-KL q"]
    betas = np.linspace(0.0, 1.0, 1001)
    for eps in (0.05, 0.1, 0.4):
        log_w, z, acc, dl, dh = anneal(path, betas, path.draw_q(runs), eps=eps)
        se = log_w.std(ddof=1) / np.sqrt(runs)
        print(f"  eps {eps:4.2f}  log Z - E log w {-log_w.mean():7.4f} +- {se:.4f}   log mean w {log_mean_exp(log_w):+7.4f}"
              f"   accept {acc:.3f}   div {dl:5d} + {dh:5d}   min v {z[:, -1].min():6.2f}")

    print("\n== reverse AIS from exact draws of p: the sandwich ==")
    for name, path in paths.items():
        for T in (10, 100, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            fwd = anneal(path, betas, path.draw_q(runs))[0]
            rev, _, acc, dl, dh = anneal(path, betas[::-1], draw_p(runs))
            lo, up = -fwd.mean(), -rev.mean()
            print(f"  {name:17s} T={T:5d}  lower {lo:8.4f}   upper {up:8.4f} +- {rev.std(ddof=1) / np.sqrt(runs):.4f}"
                  f"   width {lo + up:8.4f}   width/lower {(lo + up) / lo:6.2f}   reverse div {dl + dh:6d}")

    print("\n== reverse AIS from the forward finals instead of exact draws ==")
    for name, path in paths.items():
        for T in (100, 1000):
            betas = np.linspace(0.0, 1.0, T + 1)
            rev = anneal(path, betas[::-1], finals[(name, T)])[0]
            print(f"  {name:17s} T={T:5d}  U - log Z {-rev.mean():+8.4f} +- {rev.std(ddof=1) / np.sqrt(runs):.4f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
