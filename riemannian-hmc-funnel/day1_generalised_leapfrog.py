"""Day 1 - the generalised leapfrog on Gaussian targets.

The SMC project's day 4 needed steps about 400x smaller in the funnel's neck
than in its mouth, and no one mass matrix gives both. RMHMC makes the mass
depend on position, H = U(q) + log det G(q) / 2 + p' G(q)^-1 p / 2, and the
price is that the kinetic energy no longer separates from q, so the leapfrog
turns into Girolami and Calderhead's generalised leapfrog: an implicit half step
in p, an implicit full step in q using the mean of the two inverse metrics, and
an explicit half step in p. Before the funnel, this day checks the integrator
on a target where every answer is known: VI day 3's AR(1), r = 0.9, d = 8, with
a diagonal metric g_i(q) = m_i (1 + c q_i^2), m = diag(prec). At c = 0 that is
a constant mass and RMHMC is plain HMC. At c > 0 the target is still Gaussian
and only the kinetic energy moves.

1. At c = 0 the generalised leapfrog is the plain one, 5.3e-14 apart after 100
   steps at eps up to 0.9x plain leapfrog's stability limit of 1.42. Each
   implicit solve takes exactly 2 iterations, one to land and one to find it
   did not move, so even the degenerate case costs two gradients where plain
   HMC spends one per half step.

2. At c > 0 the fixed point converges linearly and the rate depends on eps.
   At c = 2, eps = 0.2 it needs 3.9, 6.9 and 9.9 iterations per solve for tol
   1e-4, 1e-8 and 1e-12, about 0.75 iterations a decade, and at eps = 0.71
   about 1.5 a decade. It stops converging at the step sizes plain leapfrog
   would still take: at eps = 1.42 it fails on 1 to 4 of 50 trajectories at
   c = 0.5 and on 42 or 43 at c = 2, and at 1.5x the limit it fails on nearly
   all of them. The failure is the solve, not the energy, since the steps that
   do converge at eps = 1.42 carry |dH| from 1.4 to 3.4.

3. The energy error cannot see the tolerance. Mean |dH| over 20 steps is 1.3e-2
   to 1.5e-2 at c = 2, eps = 0.2 whether tol is 1e-4 or 1e-12, and it grows as
   eps^2 as a leapfrog's should. Reversibility tracks tol instead: flip p after
   100 steps and run back, and the roundtrip misses q0 by 4x to 7x tol at the
   median from 1e-4 down to 1e-14, where plain leapfrog misses by 6.7e-15. An
   acceptance rate would look the same at tol = 1e-4 and at 1e-12 while
   the roundtrip is off by 6.5e-4, so the tolerance has to be
   set from the roundtrip and not from the acceptance.

4. With K fixed iterations per solve the step is a smooth map, and its log
   |det J| is what the volume-preservation argument says is 0. The median falls
   from 1.1e-2 at K = 1 to 6.6e-4, 3.7e-5, 1.3e-6 at K = 2, 3, 4 and reaches the
   finite-difference floor, 6e-11 against plain leapfrog's 3e-11, by K = 8. The
   worst of 20 starts is slower, 1.6e-3 at both K = 2 and K = 3 and still
   2.7e-7 at K = 8, so the median says K = 6 is plenty and the maximum says 12.

Five predictions written before the run. Three right, two half.

- Right: at c = 0 the two integrators agree to 1e-12 over 100 steps and every
  solve takes 2 iterations. 5.3e-14 and 2.00.
- Right: c = 1, eps = 0.1, tol 1e-10 needs at most 8 iterations per solve.
  It takes 6.62.
- Right: the solve stops converging below 2x plain leapfrog's stability limit.
  It fails on nearly every trajectory at 1.5x at both c.
- Half: the roundtrip error stays within 10x of tol. The median is within 4x to
  7x at every tol, the worst of 20 is 22x at 1e-12 and 28x at 1e-14.
- Half: |log det J| falls at least 10x per extra iteration. The median does, 17x
  to 29x per iteration up to K = 4, and the maximum does not move from K = 2
  to K = 3.

NumPy only. Fixed seeds. About 35 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "annealed-importance-sampling"))
from day1_ais_gaussians import ar1  # noqa: E402

RNG = np.random.default_rng(11)


class DiagMetric:
    """Gaussian target N(0, cov) with a diagonal metric g_i(q) = m_i (1 + c q_i^2).

    c = 0 is a constant mass matrix diag(m), where RMHMC is plain HMC and the
    generalised leapfrog has to reduce to the plain one. c > 0 keeps the target
    Gaussian and makes the kinetic energy depend on position, so the implicit
    solves have something to do while every other quantity stays closed form.
    """

    def __init__(self, cov, m, c):
        self.prec = np.linalg.inv(cov)
        self.m = m
        self.c = c

    def g(self, q):
        return self.m * (1.0 + self.c * q * q)

    def hamiltonian(self, q, p):
        g = self.g(q)
        return 0.5 * q @ self.prec @ q + 0.5 * np.sum(np.log(g)) + 0.5 * np.sum(p * p / g)

    def dh_dq(self, q, p):
        # dU/dq + 0.5 tr(G^-1 dG/dq_i) - 0.5 p' G^-1 dG/dq_i G^-1 p, diagonal so per coordinate
        g = self.g(q)
        dg = 2.0 * self.m * self.c * q
        return self.prec @ q + 0.5 * dg / g - 0.5 * p * p * dg / (g * g)


def fixed_point(f, x0, tol, max_iter):
    """Iterate x = f(x) from x0; returns (x, iterations, converged). tol = None runs max_iter exactly."""
    x = x0
    for k in range(1, max_iter + 1):
        x_new = f(x)
        if tol is not None and np.max(np.abs(x_new - x)) < tol:
            return x_new, k, True
        x = x_new
    return x, max_iter, tol is None


def generalised_leapfrog(model, q, p, eps, tol=1e-10, max_iter=100):
    """One step of Girolami and Calderhead's generalised leapfrog.

    Implicit half step in p, implicit full step in q with the average of the two
    inverse metrics, explicit half step in p. Returns (q, p, iterations, ok).
    """
    p_half, k1, ok1 = fixed_point(lambda ph: p - 0.5 * eps * model.dh_dq(q, ph), p, tol, max_iter)
    ginv_q = 1.0 / model.g(q)
    q_new, k2, ok2 = fixed_point(lambda qn: q + 0.5 * eps * (ginv_q + 1.0 / model.g(qn)) * p_half,
                                 q, tol, max_iter)
    p_new = p_half - 0.5 * eps * model.dh_dq(q_new, p_half)
    return q_new, p_new, k1 + k2, ok1 and ok2


def plain_leapfrog(model, q, p, eps):
    """Constant-metric leapfrog with mass diag(m)."""
    p = p - 0.5 * eps * (model.prec @ q)
    q = q + eps * p / model.m
    p = p - 0.5 * eps * (model.prec @ q)
    return q, p


def trajectory(step, q, p, n):
    iters, ok = 0, True
    for _ in range(n):
        q, p, k, good = step(q, p)
        iters += k
        ok = ok and good
    return q, p, iters, ok


def log_det_jacobian(step_map, q, p, h=1e-5):
    """log |det| of d(q', p') / d(q, p) by central differences."""
    x = np.concatenate([q, p])
    d = q.size
    jac = np.empty((2 * d, 2 * d))
    for j in range(2 * d):
        e = np.zeros(2 * d)
        e[j] = h
        plus = np.concatenate(step_map((x + e)[:d], (x + e)[d:]))
        minus = np.concatenate(step_map((x - e)[:d], (x - e)[d:]))
        jac[:, j] = (plus - minus) / (2 * h)
    return np.linalg.slogdet(jac)[1]


# ----------------------------------------------------------------------------


def main():
    np.seterr(all="ignore")  # the failing step sizes overflow on purpose
    started = time.time()
    d = 8
    cov = ar1(d, 0.9)
    prec = np.linalg.inv(cov)
    m = np.diag(prec).copy()
    stab = 2.0 / np.sqrt(np.max(np.linalg.eigvals(prec / m[:, None]).real))
    print(f"target: AR(1) r = 0.9, d = {d}, mass diag(prec) = {m[0]:.3f} .. {m[1]:.3f}"
          f"   plain leapfrog stable below eps = {stab:.4f}")
    starts = [(RNG.multivariate_normal(np.zeros(d), cov), RNG.normal(size=d) * np.sqrt(m)) for _ in range(50)]

    print("\n== c = 0: generalised against plain leapfrog, 100 steps ==")
    flat = DiagMetric(cov, m, 0.0)
    for eps in (0.05, 0.2, 0.5 * stab, 0.9 * stab):
        dq, its = 0.0, []
        for q0, p0 in starts:
            qg, pg, k, _ = trajectory(lambda q, p: generalised_leapfrog(flat, q, p, eps), q0, p0, 100)
            qp, pp = q0, p0
            for _ in range(100):
                qp, pp = plain_leapfrog(flat, qp, pp, eps)
            dq = max(dq, np.max(np.abs(qg - qp)), np.max(np.abs(pg - pp)))
            its.append(k / 200)
        print(f"  eps {eps:.4f}   max |gl - plain| {dq:.2e}   iterations per solve {np.mean(its):.2f}")

    print("\n== c > 0: iterations per implicit solve against tol, and where the solve fails ==")
    print("   c     eps     tol    iters/solve  failed steps   mean |dH| over 20 steps")
    for c in (0.5, 2.0):
        model = DiagMetric(cov, m, c)
        for eps in (0.05, 0.2, 0.5 * stab, stab, 1.5 * stab):
            for tol in (1e-4, 1e-8, 1e-12):
                its, fails, dh = [], 0, []
                for q0, p0 in starts:
                    p0 = RNG.normal(size=d) * np.sqrt(model.g(q0))
                    q, p, k, ok = trajectory(lambda q, p: generalised_leapfrog(model, q, p, eps, tol), q0, p0, 20)
                    its.append(k / 40)
                    fails += not ok
                    if ok:
                        dh.append(abs(model.hamiltonian(q, p) - model.hamiltonian(q0, p0)))
                dh_s = f"{np.mean(dh):.2e}" if dh else "   -    "
                print(f"  {c:3.1f}   {eps:.4f}  {tol:.0e}    {np.mean(its):6.2f}      {fails:3d}/50       {dh_s}")

    model = DiagMetric(cov, m, 1.0)
    its = []
    for q0, p0 in starts:
        p0 = RNG.normal(size=d) * np.sqrt(model.g(q0))
        its.append(trajectory(lambda q, p: generalised_leapfrog(model, q, p, 0.1, 1e-10), q0, p0, 20)[2] / 40)
    print(f"  1.0   0.1000  1e-10    {np.mean(its):6.2f}      (the point the prediction named)")

    print("\n== c = 2, eps = 0.2: reversibility after 100 steps, flip p and run back ==")
    model = DiagMetric(cov, m, 2.0)
    for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        errs = []
        for q0, p0 in starts[:20]:
            q, p, _, _ = trajectory(lambda q, p: generalised_leapfrog(model, q, p, 0.2, tol, 500), q0, p0, 100)
            qb, pb, _, _ = trajectory(lambda q, p: generalised_leapfrog(model, q, p, 0.2, tol, 500), q, -p, 100)
            errs.append(max(np.max(np.abs(qb - q0)), np.max(np.abs(pb + p0))))
        print(f"  tol {tol:.0e}   roundtrip error  median {np.median(errs):.2e}  max {np.max(errs):.2e}")
    errs = []
    for q0, p0 in starts[:20]:
        q, p = q0, p0
        for _ in range(100):
            q, p = plain_leapfrog(flat, q, p, 0.2)
        qb, pb = q, -p
        for _ in range(100):
            qb, pb = plain_leapfrog(flat, qb, pb, 0.2)
        errs.append(max(np.max(np.abs(qb - q0)), np.max(np.abs(pb + p0))))
    print(f"  plain c = 0     roundtrip error  median {np.median(errs):.2e}  max {np.max(errs):.2e}")

    print("\n== c = 2, eps = 0.2: log |det J| of one step with K fixed iterations per solve ==")
    for K in (1, 2, 3, 4, 6, 8, 12, 20):
        vals = [log_det_jacobian(lambda q, p: generalised_leapfrog(model, q, p, 0.2, None, K)[:2], q0, p0)
                for q0, p0 in starts[:20]]
        print(f"  K {K:2d}   median |log det J| {np.median(np.abs(vals)):.2e}   max {np.max(np.abs(vals)):.2e}")
    vals = [log_det_jacobian(lambda q, p: plain_leapfrog(flat, q, p, 0.2), q0, p0) for q0, p0 in starts[:20]]
    print(f"  plain  median |log det J| {np.median(np.abs(vals)):.2e}   max {np.max(np.abs(vals)):.2e}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
