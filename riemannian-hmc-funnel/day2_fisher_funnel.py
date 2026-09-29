"""Day 2 - Neal's funnel with the metric it has in closed form.

Day 1 measured what the implicit solves cost on a metric made up for the purpose.
This is the target the project is about, MCMC day 4's funnel, sigma_v = 3, d = 3,
with G(v) = diag(e^-v, e^-v, 1/sigma_v^2 + n/2): the negative Hessian of log p
averaged over x given v. It depends on v alone, its log determinant cancels the
n v / 2 in U, and the plan for today asked where the fixed point stops
converging on it. Centred HMC is rerun beside it from the MCMC project's own
code and seeds, 20k steps of L = 20, and cost is counted in calls to dH/dq.

1. The fixed point does not stop converging, because there is nothing implicit
   to solve. p_x's half step does not contain p, p_v's contains only p_x's, v's
   full step contains only p_v and x's only the two v's, so both solves are
   triangular. Day 1's solver takes exactly 3 iterations on each at every eps
   from 0.05 to 3.2, 4 gradients a step, and fails on 0 of 50 starts even at
   3.2 where no trajectory stays finite. Solved by hand the step is 2 gradients
   and agrees with the solver to 2.2e-15 at eps = 0.2. At eps = 0.8 and 1.6 the
   two are 2e-11 apart relative to the state, on the trajectories about to
   blow up.

2. The neck is reached at every step size. min v is -13.0, -11.9, -11.7, -13.1
   and -10.8 for eps = 0.05 to 0.8, where centred HMC gets -7.5, -6.0, -5.1,
   -4.0 and -2.7. Divergences are 0 up to eps = 0.4 against 14, 33, 266 and
   2125. Over eight seeds at eps = 0.2 and 5k steps the share of the chain below
   v = -4.16 is 8.33% +- 0.19 against p's 8.28%, and Var(v) is 8.89 +- 0.09
   against 9.00, so the variance is still about one standard error low.

3. At matched gradients the gain is all in the step size the metric allows.
   400k gradients is 10k steps by hand and 5k through the solver. At eps = 0.1
   that is an ESS in v of 685 and 353 against centred HMC's 381, so through
   the solver RMHMC is no better, 0.9x. At eps = 0.2 it is 8.1x and 4.3x. At
   eps = 0.4 IACT is 1.00 and the ESS is the chain length, 9000 and 4500,
   against 383 for centred HMC at its own best eps: 23x and 12x best against
   best. The same-eps ratio at 0.4 is 907x and says more about centred HMC's
   IACT of 1814 there than about this sampler.

4. Trajectory length matters as much as it did on a Gaussian. At eps = 0.2,
   IACT(v) is 359, 217, 47, 13, 2.7, 1.0 for L = 1, 2, 5, 10, 20, 40, and at
   L = 1 Var(v) is 4.25 with acceptance 0.998. The metric fixes the scale of a
   step and not the distance v has to travel, which is sigma_v whatever the
   metric. Sweeping sigma_v at eps = 0.2, L = 20: IACT 1.0, 1.1, 3.0, 6.0, 14.2
   for sigma_v = 1, 2, 3, 4, 6, with no divergences and acceptance 0.994
   throughout. Centred HMC over the same sweep diverges 0, 21, 113, 4204, 408
   times, so the count the MCMC project called monotone is not past sigma_v = 4.
   At 6 the chain has Var(v) at 0.39 of the truth and diverges less because it
   stays out of the neck.

5. At eps = 0.8 RMHMC diverges, 2959 of 20000 proposals, and Var(v) is still
   8.82 with the neck at 8.17% and IACT 1.47. From 20000 exact draws the rate
   rises with s = e^-v |x|^2 at the start, 6% below s = 1 and 78% above 8,
   which is the v coordinate's stiffness 1/sigma_v^2 + s/2 against a mass of
   g_v that was set from its average. That much the metric's derivation
   predicts. It does not predict that 44% of the diverged starts are below
   v = -4.16, where 8.3% of all starts are. H is invariant under a shift of v
   with x and p_x rescaled except for v^2 / 18, so the prior's slope is the
   only thing that can tell the neck from the mouth, and I have not measured
   how. The rejections cost mixing and not the neck share, which is the
   opposite of centred HMC, where a divergence marks a region the chain does
   not enter.

6. The MCMC project's divergence count adds the non-finite dH to
   nan_to_num(dH) > 1000, and nan_to_num turns inf into 1.8e308, so an
   infinite dH is counted twice. It gave 5784 for the 2959 above. On the
   centred HMC runs here it changes 8622 to 8620 and nothing else, because
   that leapfrog overflows to large finite numbers and rarely to inf. The
   count in this file takes each proposal once.

Six predictions written before the run. Four right, two half.

- Half: day 1's solver takes at most 3 iterations per solve and the step by hand
  matches it to 1e-12. 3.00 everywhere, and 2.2e-15 up to eps = 0.2, but 2.3e-11
  at eps = 0.8.
- Right: no divergences at eps up to 0.4 over 20k steps.
- Half: Var(v) within 5% of 9.00 at eps = 0.1, 0.2, 0.4. 9.08 and 8.92 are, 8.40
  at eps = 0.1 is 6.7% low with an IACT of 12.8.
- Right: min v below -9 at every eps up to 0.4. -11.7 at worst.
- Right: at 400k gradients through the solver, IACT(v) under 5 at eps = 0.4 and
  the neck share within a point of 8.3%. 1.00 and 8.62%.
- Right: acceptance under 0.8 at eps = 0.8. 0.769.

What this day does not show is RMHMC on a target whose metric is not known.
The expectation over x is what made G diagonal and the solves triangular, and
the Hessian itself has the x_i e^-v cross terms and is not positive definite
where e^-v |x|^2 / 2 is under 1/sigma_v^2 off the axis. That is day 3.

NumPy only, the chain itself on python floats. Fixed seeds. About 65 seconds.
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcmc-from-scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "annealed-importance-sampling"))
from day1_generalised_leapfrog import fixed_point  # noqa: E402
from day1_metropolis import integrated_act  # noqa: E402
from day3_hmc import Funnel, hmc  # noqa: E402
from day4_diagnostics import divergences as divergences_mcmc  # noqa: E402

NECK = -4.16
BURN = 0.1
BLOWUP = (OverflowError, ZeroDivisionError)  # e^v past the float range, either way


class FisherFunnel:
    """The funnel (x_1..x_n, v) with G(v) = diag(e^-v, .., e^-v, 1/sigma_v^2 + n/2).

    G is the negative Hessian of log p averaged over x given v. The cross terms
    x_i e^-v average to 0 and the v term 1/sigma^2 + e^-v sum x^2 / 2 averages to
    1/sigma^2 + n/2, so the metric is diagonal and a function of v alone. Its log
    determinant is -n v plus a constant, which cancels the n v / 2 in U exactly:

        H = v^2 / (2 sigma^2) + e^-v |x|^2 / 2 + e^v |p_x|^2 / 2 + p_v^2 / (2 g_v)
    """

    def __init__(self, d=3, sigma_v=3.0):
        self.d, self.n, self.sigma_v = d, d - 1, sigma_v
        self.gv = 1.0 / sigma_v**2 + 0.5 * (d - 1)

    def g(self, q):
        out = np.full(self.d, np.exp(-q[-1]))
        out[-1] = self.gv
        return out

    def hamiltonian(self, q, p):
        v = q[-1]
        return (0.5 * (v / self.sigma_v) ** 2 + 0.5 * np.exp(-v) * np.dot(q[:-1], q[:-1])
                + 0.5 * np.exp(v) * np.dot(p[:-1], p[:-1]) + 0.5 * p[-1] ** 2 / self.gv)

    def dh_dq(self, q, p):
        v = q[-1]
        out = np.empty(self.d)
        out[:-1] = np.exp(-v) * q[:-1]
        out[-1] = (v / self.sigma_v**2 - 0.5 * np.exp(-v) * np.dot(q[:-1], q[:-1])
                   + 0.5 * np.exp(v) * np.dot(p[:-1], p[:-1]))
        return out


def generic_step(model, q, p, eps, tol=1e-12, max_iter=100):
    """Day 1's generalised leapfrog, with the dH/dq calls counted.

    Returns (q, p, momentum iterations, position iterations, gradient calls, ok).
    """
    calls = [0]

    def grad(qq, pp):
        calls[0] += 1
        return model.dh_dq(qq, pp)

    p_half, k1, ok1 = fixed_point(lambda ph: p - 0.5 * eps * grad(q, ph), p, tol, max_iter)
    ginv = 1.0 / model.g(q)
    q_new, k2, ok2 = fixed_point(lambda qn: q + 0.5 * eps * (ginv + 1.0 / model.g(qn)) * p_half,
                                 q, tol, max_iter)
    p_new = p_half - 0.5 * eps * grad(q_new, p_half)
    return q_new, p_new, k1, k2, calls[0], ok1 and ok2


def explicit_step(x, v, px, pv, eps, s2inv, gv):
    """The same step solved by hand, two gradients, on python floats.

    p_x's half step does not contain p. p_v's contains only p_x's, which is by
    then known. v's full step contains only p_v, and x's only the two v's. So the
    two implicit solves are triangular and each is one substitution.
    """
    ev = math.exp(-v)
    px = [a - 0.5 * eps * ev * b for a, b in zip(px, x)]
    pp = sum(a * a for a in px)
    pv = pv - 0.5 * eps * (v * s2inv - 0.5 * ev * sum(b * b for b in x) + 0.5 * pp / ev)
    v_new = v + eps * pv / gv
    scale = 0.5 * eps * (1.0 / ev + math.exp(v_new))
    x = [b + scale * a for a, b in zip(px, x)]
    ev = math.exp(-v_new)
    pv = pv - 0.5 * eps * (v_new * s2inv - 0.5 * ev * sum(b * b for b in x) + 0.5 * pp / ev)
    px = [a - 0.5 * eps * ev * b for a, b in zip(px, x)]
    return x, v_new, px, pv


def energy(x, v, px, pv, s2inv, gv):
    return (0.5 * v * v * s2inv + 0.5 * math.exp(-v) * sum(b * b for b in x)
            + 0.5 * math.exp(v) * sum(a * a for a in px) + 0.5 * pv * pv / gv)


def rmhmc(model, n_steps, eps, n_leap, rng):
    """RMHMC with p ~ N(0, G(q)) redrawn every step. Returns chain, acceptance, dH."""
    s2inv, gv, n = 1.0 / model.sigma_v**2, model.gv, model.n
    x, v = [0.0] * n, 0.0
    chain = np.empty((n_steps, n + 1))
    dH = np.empty(n_steps)
    noise = rng.standard_normal((n_steps, n + 1))
    unif = np.log(rng.random(n_steps))
    acc = 0
    for t in range(n_steps):
        sd = math.exp(-0.5 * v)
        px = [sd * z for z in noise[t, :n]]
        pv = math.sqrt(gv) * noise[t, n]
        h0 = energy(x, v, px, pv, s2inv, gv)
        xn, vn, pxn, pvn = x, v, px, pv
        try:
            for _ in range(n_leap):
                xn, vn, pxn, pvn = explicit_step(xn, vn, pxn, pvn, eps, s2inv, gv)
            delta = energy(xn, vn, pxn, pvn, s2inv, gv) - h0
        except BLOWUP:
            delta = math.inf
        dH[t] = delta
        if math.isfinite(delta) and unif[t] < -delta:
            x, v = xn, vn
            acc += 1
        chain[t, :n], chain[t, n] = x, v
    return chain, acc / n_steps, dH


def divergences(dH, threshold=1000.0):
    """Proposals whose energy error is not finite or is past the threshold, each counted once.

    The MCMC project's count adds the non-finite ones to nan_to_num(dH) > threshold,
    and nan_to_num turns inf into 1.8e308, so every infinite dH is in both terms.
    """
    dH = np.asarray(dH, float)
    return int(np.sum(~np.isfinite(dH) | (np.nan_to_num(dH, nan=0.0, posinf=0.0, neginf=0.0) > threshold)))


def report(label, chain, acc, dH, old_count=False):
    v = chain[int(BURN * len(chain)):, -1]
    x0 = chain[int(BURN * len(chain)):, 0]
    print(f"  {label}  acc {acc:.3f}  div {divergences(dH):5d}  Var(v) {v.var():6.2f}"
          f"  min v {v.min():7.2f}  max v {v.max():6.2f}  neck {100 * np.mean(v < NECK):5.2f}%"
          f"  IACT(v) {integrated_act(v):7.2f}  Var(x0) {x0.var():8.2f}")
    if old_count:
        print(f"            the mcmc project's count: {divergences_mcmc(dH)}, with"
              f" {int(np.sum(np.isinf(dH)))} infinite dH in it twice")
    return integrated_act(v)


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ----------------------------------------------------------------------------


def main():
    np.seterr(all="ignore")
    started = time.time()
    d = 3
    model = FisherFunnel(d)
    rng = np.random.default_rng(22)
    print(f"funnel d = {d}, sigma_v = 3: Var(v) = 9.00, Var(x) = {math.exp(4.5):.2f},"
          f" P(v < {NECK}) = {100 * phi(NECK / 3.0):.2f}%, g_v = {model.gv:.4f}")

    print("\n== the generic solver from day 1 on this metric, 50 draws from p, 20 steps each ==")
    print("    eps   p iters  q iters  grads/step  failed   max |generic - explicit|   median |dH|")
    starts = []
    for _ in range(50):
        v = 3.0 * rng.standard_normal()
        q = np.append(np.exp(0.5 * v) * rng.standard_normal(d - 1), v)
        starts.append((q, np.sqrt(model.g(q)) * rng.standard_normal(d)))
    for eps in (0.05, 0.2, 0.8, 1.6, 3.2):
        k1s, k2s, grads, fails, gap, dh = [], [], [], 0, 0.0, []
        for q0, p0 in starts:
            q, p = q0, p0
            x, v, px, pv = list(q0[:-1]), q0[-1], list(p0[:-1]), p0[-1]
            good = True
            for _ in range(20):
                q, p, k1, k2, g, ok = generic_step(model, q, p, eps)
                try:
                    x, v, px, pv = explicit_step(x, v, px, pv, eps, 1.0 / 9.0, model.gv)
                except BLOWUP:
                    good = False
                    break
                if not (ok and np.isfinite(q).all()):
                    fails += bool(not ok and np.isfinite(q).all() and np.isfinite(p).all())
                    good = False
                    break
                k1s.append(k1), k2s.append(k2), grads.append(g)
                scale = max(1.0, np.max(np.abs(q)), np.max(np.abs(p)))
                gap = max(gap, np.max(np.abs(q - np.append(x, v))) / scale,
                          np.max(np.abs(p - np.append(px, pv))) / scale)
            if good:
                dh.append(abs(model.hamiltonian(q, p) - model.hamiltonian(q0, p0)))
        print(f"   {eps:4.2f}   {np.mean(k1s):5.2f}    {np.mean(k2s):5.2f}     {np.mean(grads):5.2f}"
              f"     {fails:3d}/50        {gap:.2e}            {np.median(dh) if dh else np.nan:.2e}"
              f"   ({len(dh)} finite)")

    print("\n== centred hmc from the mcmc project, 20k steps, L = 20, 400k gradients ==")
    fun = Funnel(d)
    base = {}
    for eps in (0.05, 0.1, 0.2, 0.4, 0.8):
        ch, acc, dH, _ = hmc(fun, np.zeros(d), 20_000, eps, 20, np.random.default_rng(11))
        base[eps] = report(f"eps {eps:4.2f}", ch, acc, dH, old_count=True)

    print("\n== rmhmc, fisher metric, 20k steps, L = 20: 800k gradients solved by hand ==")
    for eps in (0.05, 0.1, 0.2, 0.4, 0.8):
        ch, acc, dH = rmhmc(model, 20_000, eps, 20, np.random.default_rng(11))
        report(f"eps {eps:4.2f}", ch, acc, dH)

    print("\n== rmhmc at 400k gradients: by hand 10k steps, by the generic solver's 4 a step 5k ==")
    for n_steps, tag in ((10_000, "by hand"), (5_000, "generic")):
        for eps in (0.1, 0.2, 0.4, 0.8):
            ch, acc, dH = rmhmc(model, n_steps, eps, 20, np.random.default_rng(11))
            iact = report(f"{tag} n {n_steps:5d} eps {eps:4.2f}", ch, acc, dH)
            ess = 0.9 * n_steps / iact
            ess_b = 0.9 * 20_000 / base[eps]
            print(f"      ess(v) {ess:7.0f} against centred hmc's {ess_b:6.0f} at the same eps, {ess / ess_b:6.1f}x")

    print("\n== trajectory length at eps = 0.2, 5k steps ==")
    for n_leap in (1, 2, 5, 10, 20, 40, 80):
        ch, acc, dH = rmhmc(model, 5_000, 0.2, n_leap, np.random.default_rng(11))
        report(f"L {n_leap:3d}", ch, acc, dH)

    print("\n== eight seeds at eps = 0.2, L = 20, 5k steps: the neck share and its spread ==")
    shares, vars_, mins = [], [], []
    for seed in range(8):
        ch, acc, dH = rmhmc(model, 5_000, 0.2, 20, np.random.default_rng(100 + seed))
        v = ch[500:, -1]
        shares.append(100 * np.mean(v < NECK)), vars_.append(v.var()), mins.append(v.min())
    print(f"  neck {np.mean(shares):.2f}% +- {np.std(shares, ddof=1) / np.sqrt(8):.2f}"
          f"   Var(v) {np.mean(vars_):.2f} +- {np.std(vars_, ddof=1) / np.sqrt(8):.2f}"
          f"   min v {np.min(mins):.2f} to {np.max(mins):.2f}")

    print("\n== neck depth: sigma_v swept at eps = 0.2, L = 20, 12k steps ==")
    for sig in (1.0, 2.0, 3.0, 4.0, 6.0):
        ch, acc, dH, _ = hmc(Funnel(d, sigma_v=sig), np.zeros(d), 12_000, 0.2, 20, np.random.default_rng(31))
        v = ch[1500:, -1]
        m = FisherFunnel(d, sigma_v=sig)
        ch2, acc2, dH2 = rmhmc(m, 12_000, 0.2, 20, np.random.default_rng(31))
        v2 = ch2[1500:, -1]
        print(f"  sigma_v {sig:3.1f}   hmc   ratio {v.var() / sig**2:5.3f}  div {divergences(dH):5d} ({divergences_mcmc(dH):5d})"
              f"  acc {acc:.3f}   |   rmhmc ratio {v2.var() / sig**2:5.3f}  div {divergences(dH2):5d}"
              f"  acc {acc2:.3f}  min v / sigma {v2.min() / sig:6.2f}  IACT {integrated_act(v2):6.2f}")

    print("\n== where the energy error comes from: |dH| of 20 steps at eps = 0.8 by s = e^-v |x|^2 at the start ==")
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(20_000):
        v = 3.0 * rng.standard_normal()
        x = list(math.exp(0.5 * v) * rng.standard_normal(d - 1))
        px = list(math.exp(-0.5 * v) * rng.standard_normal(d - 1))
        pv = math.sqrt(model.gv) * rng.standard_normal()
        s = math.exp(-v) * sum(b * b for b in x)
        h0 = energy(x, v, px, pv, 1.0 / 9.0, model.gv)
        smax = s
        try:
            for _ in range(20):
                x, v, px, pv = explicit_step(x, v, px, pv, 0.8, 1.0 / 9.0, model.gv)
                smax = max(smax, math.exp(-v) * sum(b * b for b in x))
            dh = abs(energy(x, v, px, pv, 1.0 / 9.0, model.gv) - h0)
        except BLOWUP:
            dh = math.inf
        rows.append((s, smax, dh, v))
    rows = np.array(rows)
    dh = np.where(np.isfinite(rows[:, 2]), rows[:, 2], np.inf)
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 100)):
        sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        print(f"  s in [{lo:2d}, {hi:3d})  n {sel.sum():5d}   median |dH| {np.median(dh[sel]):.3f}"
              f"   |dH| > 1: {100 * np.mean(dh[sel] > 1):5.1f}%   > 1000: {100 * np.mean(dh[sel] > 1000):5.2f}%")
    bad = dh > 1000
    print(f"  diverged {bad.sum()} of 20000; of those, start v < {NECK}: {100 * np.mean(rows[bad, 3] < NECK):.1f}%"
          f" against {100 * phi(NECK / 3.0):.1f}% of all starts")
    print(f"  corr(log |dH|, v at start) = {np.corrcoef(np.log(dh[~bad] + 1e-300), rows[~bad, 3])[0, 1]:+.3f}"
          f"   corr(log |dH|, log s) = {np.corrcoef(np.log(dh[~bad] + 1e-300), np.log(rows[~bad, 0]))[0, 1]:+.3f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
