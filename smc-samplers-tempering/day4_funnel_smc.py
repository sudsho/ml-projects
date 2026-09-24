"""Day 4 - Neal's funnel again, SMC against AIS at the same gradient cost.

AIS day 4 left the funnel (sigma_v = 3, d = 3, log Z = 0) with `log mean w`
about 0.09 low at every T from 10 to 1000, from VI day 4's reverse-KL q, and
blamed it on runs that never reached the neck: p has 8.2% of its mass below
v = -4.16 and the AIS finals had 4.75% at best. Same q, same funnel, same HMC
move (eps 0.2, 10 leapfrog steps, MCMC day 4's divergence definition), imported
from that file. SMC here is day 2's adaptive rule, resampling at every step,
with 1, 5 or 20 HMC moves per temperature, and day 3's idea of a kernel shaped
by the cloud as a second move. Cost is counted in HMC moves per particle, each
ten gradients, and AIS is rerun beside it at N = 2000 with 8 repeats per row,
so its `log mean w` has a standard error this time.

1. At the same cost SMC gets further, and it still stops short. At rho = 0.9
   and 20 moves, 168 moves per particle, SMC is 0.068 +- 0.008 low. AIS at
   T = 300, 299 moves, is 0.094 +- 0.006 low, and at T = 100 0.110 +- 0.017.
   At 41 moves, rho = 0.9 and 5 moves, SMC is 0.090 low, already level with
   AIS at seven times the cost. So resampling does move the plateau, by about a
   quarter of it, and the curve is still flattening at the last row.

2. At a small budget it is the other way round. rho = 0.5 with one move is 3
   moves per particle and 0.264 low, against 0.176 for AIS at T = 10. Four
   temperatures and one move each is a resample onto a cloud that never gets
   to spread, which is day 3's section 1 on a target without a closed form.

3. The cloud goes into the neck, partly. P(v < -4.16) at the end is 5.2% for
   the best SMC row against 3.5% for AIS at T = 300 and p's 8.2%, min v -5.67
   against -5.49. That is 63% of the neck mass where AIS had 43%, with fewer
   moves. The missing third is the 0.068.

4. Resampling rarely happens and the genealogy barely narrows. The adaptive
   rule takes 3.6 to 9.6 temperatures, so the final cloud has 1022 to 1152
   distinct starting ancestors of 2000. One schedule at rho = 0.9 is 0, 0.317,
   0.568, 0.732, 0.829, 0.899, 0.945, 0.976, 0.999, 1, with three of its eight
   move blocks past beta 0.9 where AIS's uniform path has a tenth. That is
   where the divergences are, 1390 of 1461 in the best row, and SMC diverges
   12x as often per move as AIS at T = 300, because more of its particles are
   standing in the neck when the path opens it.

5. The cloud-shaped kernel is the wrong kernel here. Inverse mass set to the
   cloud's variances, step 0.5: at rho = 0.9 and 20 moves it is 0.112 low with
   75209 divergences, and with one move almost nothing reaches the neck, 0.2%.
   The cloud's x variance is the mouth's, so the steps are too big wherever v is
   small. At rho = 0.5 and 20 moves its mean, 0.113 low, is less biased than the
   fixed kernel's 0.155, with an sd of 0.20 against 0.043.

Five predictions written before the run. Two right, one half, two wrong.

- Right: the best SMC row puts more than 5% of the cloud below v = -4.16.
  5.16%, by a margin I would not lean on.
- Right: most of the fixed kernel's divergences past beta 0.9. 95% in the best
  row.
- Half: the cloud-shaped kernel more biased than the fixed one at 20 moves. At
  rho = 0.9, 0.112 against 0.068. At rho = 0.5 less biased on the mean, 4.7x
  noisier.
- Wrong: SMC within 0.05 of log Z at rho = 0.9 and 20 moves. 0.068.
- Wrong: under 10% of starting ancestors left at rho = 0.5. 51% to 56%. With
  four resamplings in the whole run there is no depth for the lineages to
  collapse into, which I should have read off day 2's temperature counts.

NumPy only. Fixed seeds. About 30 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "annealed-importance-sampling"))
sys.path.insert(0, HERE)
import day4_funnel_ais as ais4  # noqa: E402
from day2_adaptive_tempering import log_sum_exp, next_beta_sample  # noqa: E402
import day1_tempered_smc as day1  # noqa: E402

RNG = np.random.default_rng(44)
ais4.RNG = RNG
day1.RNG = RNG

NECK = -4.16


# ----------------------------------------------------------------------------
# A move tuned from the cloud
# ----------------------------------------------------------------------------


def cloud_hmc_step(path, beta, z, eps, n_leap, threshold=1000.0):
    """HMC with a diagonal mass matrix read off the cloud, inverse mass = the cloud's variances.

    Day 3's lesson on the Gaussians was that a kernel shaped like the cloud beats
    any number of moves of one that is not. On the funnel the cloud's variance in
    x is set by the mouth, where x is wide, and the neck needs steps 400x smaller,
    so a single global shape is the wrong one for most of the mass at some beta.
    Same divergence definition as ais4.hmc_step.
    """
    s2 = z.var(axis=0) + 1e-12
    with np.errstate(over="ignore", invalid="ignore"):
        p0 = RNG.standard_normal(z.shape) / np.sqrt(s2)
        h0 = -path.log_f(beta, z) + 0.5 * np.sum(p0 * p0 * s2, axis=1)
        q, p = z.copy(), p0 + 0.5 * eps * path.grad(beta, z)
        for i in range(n_leap):
            q = q + eps * s2 * p
            g = path.grad(beta, q)
            p = p + (eps if i < n_leap - 1 else 0.5 * eps) * g
        h1 = -path.log_f(beta, q) + 0.5 * np.sum(p * p * s2, axis=1)
        dh = h1 - h0
    bad = ~np.isfinite(dh)
    diverged = bad | (np.nan_to_num(dh, nan=0.0) > threshold)
    dh = np.where(bad, np.inf, dh)
    accept = np.log(RNG.random(len(z))) < -dh
    return np.where(accept[:, None], q, z), accept, diverged


KERNELS = {"hmc 0.2": (ais4.hmc_step, 0.2), "cloud hmc": (cloud_hmc_step, 0.5)}


# ----------------------------------------------------------------------------
# The sampler
# ----------------------------------------------------------------------------


def smc(path, n, rho, moves, kernel):
    """Adaptive tempered SMC, resampling at every step, `moves` HMC moves per temperature.

    Returns log Z_hat, the number of temperatures, the final cloud, the
    divergences at beta <= 0.9 and above, the moves per particle, and the
    number of distinct starting ancestors.
    """
    step, eps = KERNELS[kernel]
    z = path.draw_q(n)
    ancestor = np.arange(n)
    beta, log_z, temps, spent = 0.0, 0.0, 0, 0
    div_low = div_high = 0
    while beta < 1.0:
        g = ais4.funnel_log_p(z, ais4.SIGMA_V) - path.log_q(z)
        nxt = next_beta_sample(g, beta, rho)
        inc = (nxt - beta) * g
        log_z += log_sum_exp(inc) - np.log(n)
        w = np.exp(inc - inc.max())
        idx = day1.systematic(w / w.sum())
        z, ancestor, beta = z[idx], ancestor[idx], nxt
        temps += 1
        if beta >= 1.0:
            break
        for _ in range(moves):
            z, _, dv = step(path, beta, z, eps, 10)
            spent += 1
            if beta > 0.9:
                div_high += int(dv.sum())
            else:
                div_low += int(dv.sum())
    return log_z, temps, z, div_low, div_high, spent, len(np.unique(ancestor))


def ais(path, n, T):
    """AIS day 4's sampler, uniform schedule, one HMC move per interior temperature, log mean w."""
    log_w, z, _, dl, dh = ais4.anneal(path, np.linspace(0.0, 1.0, T + 1), path.draw_q(n))
    return ais4.log_mean_exp(log_w), z, dl, dh


# ----------------------------------------------------------------------------


def neck(z):
    v = z[:, -1]
    return np.mean(v < NECK), v.min()


def main():
    started = time.time()
    n, reps = 2000, 8
    path = ais4.FunnelPath(*ais4.mean_field_optimum(ais4.SIGMA_V, ais4.D))
    ref = ais4.draw_p(400000)[:, -1]
    print(f"funnel sigma_v = {ais4.SIGMA_V}, d = {ais4.D}, log Z = 0, reverse-KL q"
          f"   p: P(v < {NECK}) {np.mean(ref < NECK):.4f}   N = {n}, {reps} runs per row")

    print("\n  sampler             rho  moves    T    moves/particle   log Z_hat         sd"
          "     P(v<-4.16)   min v    div <=0.9   >0.9   ancestors")
    for kernel in KERNELS:
        for rho in (0.5, 0.9):
            for moves in (1, 5, 20):
                rows = [smc(path, n, rho, moves, kernel) for _ in range(reps)]
                est = np.array([r[0] for r in rows])
                nk = np.array([neck(r[2]) for r in rows])
                spent = np.mean([r[5] for r in rows])
                print(f"  smc {kernel:10s}   {rho:4.2f}  {moves:4d}  {np.mean([r[1] for r in rows]):5.1f}"
                      f"   {spent:8.1f}        {est.mean():+.4f} +- {est.std(ddof=1) / np.sqrt(reps):.4f}"
                      f"  {est.std(ddof=1):.4f}   {nk[:, 0].mean():.4f}   {nk[:, 1].min():6.2f}"
                      f"   {np.mean([r[3] for r in rows]):8.0f}  {np.mean([r[4] for r in rows]):6.0f}"
                      f"   {np.mean([r[6] for r in rows]):7.1f}")

    print("\n  AIS at matched moves per particle, same N")
    for T in (10, 30, 100, 300):
        rows = [ais(path, n, T) for _ in range(reps)]
        est = np.array([r[0] for r in rows])
        nk = np.array([neck(r[1]) for r in rows])
        print(f"  ais hmc 0.2          -     1  {T:5d}   {T - 1:8.1f}        {est.mean():+.4f} +- "
              f"{est.std(ddof=1) / np.sqrt(reps):.4f}  {est.std(ddof=1):.4f}   {nk[:, 0].mean():.4f}"
              f"   {nk[:, 1].min():6.2f}   {np.mean([r[2] for r in rows]):8.0f}  {np.mean([r[3] for r in rows]):6.0f}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
