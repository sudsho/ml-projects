"""Day 4 - RMHMC as the move step inside SMC day 4's tempered sampler.

SMC day 4 left the funnel with 63% of p's neck mass below v = -4.16 at 168
HMC moves per particle, 0.068 low in log Z, and 75209 divergences when the
kernel was shaped like the cloud, because the cloud's x variance is the
mouth's. Day 2 found that the Fisher metric reaches the neck at every step
size with no divergences up to eps = 0.4, and day 3 found that SoftAbs does
not survive its own solver. So the move step here is day 2's sampler on the
tempered target f_beta = q^(1 - beta) p^beta with the metric that target has
in closed form, G_beta(v) = beta G_p(v) + (1 - beta) Prec_q, which is diagonal
because VI day 4's q is mean-field, and a function of v alone, so both
implicit solves are still triangular and the step is two gradients, explicit,
vectorised over the cloud. Prec_q's v entry is 1/0.9 and G_p's is
1/9 + 1, both 1.111, so the v mass does not move with beta at all. Same
sampler, schedule rule, resampler, q, N = 2000, 8 runs per row, and the same
divergence definition, with AIS day 4's HMC kernel rerun beside it from the
same seed. Cost is counted in gradients per particle, two per leapfrog step
here against one there. SoftAbs is left out on day 3's evidence.

1. At beta = 1 the step is day 2's, 1.6e-15 apart in the state and 2.0e-14 in
   dH after 20 steps, and the roundtrip, flip p and run back, misses the start
   by 1e-16 at the median at every beta. It is not exact everywhere: from
   draws of p, 3 of 200 trajectories at beta = 0.3 and 2 at 0.7 are lost at
   eps = 0.2, 19 and 9 at eps = 0.4, and 0 at beta = 0 and 1. Every lost one
   starts in the mouth, v = 5.8 to 8.7 at eps = 0.2, and runs v off to +1e12,
   not into the neck. A draw of p at v = 8 has |x| about e^4, and f_0.3 gives
   x a precision floor of 0.7 x 1.57, so those are 50-sigma starts the
   sampler never stands on, and the sampler's rows diverge 0 times at
   eps = 0.2 and 0.4 at every temperature.

2. One RMHMC move per temperature fills the neck. At rho = 0.9 the cloud's
   share below v = -4.16 is 8.54%, 8.54% and 8.46% at 1, 5 and 20 moves,
   against p's 8.25% and the HMC move's 4.68% at 5 moves, 6.45% at 20 and
   5.33% at 40, which is the same gradients as RMHMC at 20. min v is -11 to
   -13 against -5.6, P(v < -7) is 1.1% against p's 0.98% and HMC's 0.00%,
   and Var(v) is 8.42 against 9.00 and HMC's 6.87. Acceptance is 0.995 at
   every temperature, the neck fills in the last tenth of the path, 2.7% at
   beta = 0.90, 6.3% at 0.98 and 8.4% at 0.994, and min v reaches -7 by
   beta = 0.83 where the HMC cloud never passes -5.3.

3. The moves past the first buy log Z and not much of it. The RMHMC rows are
   0.042 +- 0.018, 0.036 +- 0.011 and 0.033 +- 0.012 low at 1, 5 and 20
   moves at eps = 0.2, and 0.020 and 0.024 at eps = 0.4. The HMC rows are
   0.109, 0.051 and 0.064 low, the 20-move one with an sd of 0.085 from one
   bad run, and AIS at T = 300 is 0.060 low. With the neck at p's share and
   the estimate still 0.02 to 0.04 low, what is left of the SMC project's
   0.068 is its day 2's adaptivity bias, which this kernel cannot touch. The
   schedule takes 9.6 to 10 temperatures and 1050 to 1115 of 2000 starting
   ancestors survive in every row, RMHMC or not.

4. At eps = 0.8 the sampler diverges 17888 times in a run of 20 moves, 2968
   at beta <= 0.9 and 14920 past it, at acceptance 0.80 to 0.90, and the neck
   share is still 8.26% with log Z 0.039 low. Day 2's finding again: with
   this metric a divergence costs mixing and not the neck, where for the HMC
   move 2246 divergences go with a cloud that has two thirds of it.

5. The cost. With the neck share's implied ESS, p(1 - p) / Var(p_hat) over
   8 runs, RMHMC at one move is 87 gradients per effective sample at 172
   gradients a particle, against day 2's chain at 44, and the best HMC row
   is 445 at rho = 0.5 with 3.2% of the neck. At 20 moves it is 2955,
   because the ESS cannot pass N and the gradients keep coming. The ESS
   column carries about +-50% itself at 8 runs.

Seven predictions written before the run. Four right, one half, two wrong.

- Half: the step matches day 2's to 1e-12 and the roundtrip misses by under
  1e-10 at every beta. 1.6e-15 and 1e-16 at the median, and 3 to 19 of 200
  trajectories lost at beta = 0.3 and 0.7 from p's mouth.
- Right: rmhmc 0.2 at rho = 0.9 and 20 moves puts at least 7.5% below
  v = -4.16, against hmc's 5.2%. 8.46%, against 6.45% in this run's seed.
- Wrong: that row within 0.03 of log Z. 0.033 +- 0.012, and 0.024 at
  eps = 0.4.
- Right: that row diverges under 100 times and eps = 0.8 over 2000. 0 and
  17888.
- Right: hmc at 40 moves still under 6.5%. 5.33%.
- Right: the implied ESS of the best rmhmc row at least 500. 2369, and 3986
  at one move.
- Wrong: at rho = 0.5 and 1 move rmhmc still more than 0.15 low and no better
  than hmc, because the fault there is the resample. 0.145 against 0.232, and
  7.2% of the neck against 3.2% at 60 gradients, so the kernel is most of
  that fault too.

NumPy only. Fixed seeds. About 100 seconds.
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "smc-samplers-tempering"))
sys.path.insert(0, os.path.join(HERE, "..", "annealed-importance-sampling"))
sys.path.insert(0, HERE)
import day4_funnel_smc as smc4  # noqa: E402
import day4_funnel_ais as ais4  # noqa: E402
import day1_tempered_smc as smc1  # noqa: E402
from day2_adaptive_tempering import log_sum_exp, next_beta_sample  # noqa: E402
from day2_fisher_funnel import NECK, FisherFunnel, energy, explicit_step  # noqa: E402

RNG = np.random.default_rng(45)
smc4.RNG = ais4.RNG = smc1.RNG = RNG
D, SIGMA_V = ais4.D, ais4.SIGMA_V


class TemperedFisher:
    """The funnel path's metric in closed form: G_beta(v) = beta G_p(v) + (1 - beta) Prec_q.

    G_p is day 2's, the Hessian of -log p averaged over x given v, and the
    Hessian of -log q is its precision, so the average of the tempered target's
    Hessian is the same convex combination. With a mean-field q both are
    diagonal: m_x(v) = beta e^-v + (1 - beta) a_x and m_v = beta g_v + (1 - beta) a_v.
    H = U_beta + p' G^-1 p / 2 + log det G / 2, and the only term of dH/dq that
    holds p is dH/dv through m_x(v), so the generalised leapfrog is explicit:
    p_x's half step, then p_v's from it, then v's full step, then x's from the
    two v's. At beta = 1 it is day 2's step and at beta = 0 plain HMC with
    mass Prec_q.
    """

    def __init__(self, path):
        prec = np.diag(path.prec)
        assert np.allclose(path.prec, np.diag(prec)), "the metric is diagonal only for a mean-field q"
        self.path = path
        self.ax, self.av = prec[:-1], prec[-1]
        self.gv = 1.0 / SIGMA_V**2 + 0.5 * (D - 1)

    def mx(self, beta, v):
        return beta * np.exp(-v)[:, None] + (1.0 - beta) * self.ax[None, :]

    def mv(self, beta):
        return beta * self.gv + (1.0 - beta) * self.av

    def extra(self, beta, v, px):
        """The part of dH/dv that the metric adds: (log det G)' / 2 - p_x' (G^-1)' p_x / 2."""
        mx = self.mx(beta, v)
        dm = -beta * np.exp(-v)[:, None]
        return 0.5 * np.sum(dm / mx - px * px * dm / (mx * mx), axis=1)

    def hamiltonian(self, beta, z, p):
        mx, mv = self.mx(beta, z[:, -1]), self.mv(beta)
        return (-self.path.log_f(beta, z) + 0.5 * np.sum(p[:, :-1] ** 2 / mx, axis=1) + 0.5 * p[:, -1] ** 2 / mv
                + 0.5 * np.sum(np.log(mx), axis=1) + 0.5 * math.log(mv))

    def draw_p(self, beta, z):
        p = RNG.standard_normal(z.shape)
        p[:, :-1] *= np.sqrt(self.mx(beta, z[:, -1]))
        p[:, -1] *= math.sqrt(self.mv(beta))
        return p

    def leap(self, beta, q, p, eps):
        x, v = q[:, :-1], q[:, -1]
        g = -self.path.grad(beta, q)
        px = p[:, :-1] - 0.5 * eps * g[:, :-1]
        pv = p[:, -1] - 0.5 * eps * (g[:, -1] + self.extra(beta, v, px))
        v_new = v + eps * pv / self.mv(beta)
        x_new = x + 0.5 * eps * (1.0 / self.mx(beta, v) + 1.0 / self.mx(beta, v_new)) * px
        q_new = np.column_stack([x_new, v_new])
        g = -self.path.grad(beta, q_new)
        pv = pv - 0.5 * eps * (g[:, -1] + self.extra(beta, v_new, px))
        px = px - 0.5 * eps * g[:, :-1]
        return q_new, np.column_stack([px, pv])


MODEL = [None]


def rmhmc_step(path, beta, z, eps, n_leap, threshold=1000.0):
    """One RMHMC move at temperature beta for the whole cloud. Same return and divergence rule as ais4.hmc_step."""
    model = MODEL[0]
    with np.errstate(over="ignore", invalid="ignore"):
        p0 = model.draw_p(beta, z)
        h0 = model.hamiltonian(beta, z, p0)
        q, p = z, p0
        for _ in range(n_leap):
            q, p = model.leap(beta, q, p, eps)
        dh = model.hamiltonian(beta, q, p) - h0
    bad = ~np.isfinite(dh)
    diverged = bad | (np.nan_to_num(dh, nan=0.0) > threshold)
    dh = np.where(bad, np.inf, dh)
    accept = np.log(RNG.random(len(z))) < -dh
    return np.where(accept[:, None], q, z), accept, diverged


GRADS = {"hmc 0.2": 10, "rmhmc 0.2": 20, "rmhmc 0.4": 20, "rmhmc 0.8": 20}


def smc_trace(path, n, rho, moves, kernel):
    """smc4.smc with one line per temperature: beta, acceptance, divergences and the neck share after the moves."""
    step, eps = smc4.KERNELS[kernel]
    z = path.draw_q(n)
    beta, log_z = 0.0, 0.0
    print(f"      beta    acc   div   P(v<-4.16)   min v   mean v")
    while beta < 1.0:
        g = ais4.funnel_log_p(z, SIGMA_V) - path.log_q(z)
        nxt = next_beta_sample(g, beta, rho)
        inc = (nxt - beta) * g
        log_z += log_sum_exp(inc) - np.log(n)
        w = np.exp(inc - inc.max())
        z, beta = z[smc1.systematic(w / w.sum())], nxt
        if beta >= 1.0:
            break
        acc = div = 0
        for _ in range(moves):
            z, a, dv = step(path, beta, z, eps, 10)
            acc += int(a.sum())
            div += int(dv.sum())
        v = z[:, -1]
        print(f"    {beta:6.3f}  {acc / (moves * n):.3f}  {div:4d}      {np.mean(v < NECK):.4f}   {v.min():6.2f}   {v.mean():6.2f}")
    print(f"    log Z_hat {log_z:+.4f}")


def main():
    started = time.time()
    n, reps = 2000, 8
    path = ais4.FunnelPath(*ais4.mean_field_optimum(SIGMA_V, D))
    model = TemperedFisher(path)
    MODEL[0] = model
    smc4.KERNELS.update({"rmhmc 0.2": (rmhmc_step, 0.2), "rmhmc 0.4": (rmhmc_step, 0.4), "rmhmc 0.8": (rmhmc_step, 0.8)})
    ref = ais4.draw_p(400000)[:, -1]
    p_neck = np.mean(ref < NECK)
    print(f"funnel sigma_v = {SIGMA_V}, d = {D}, log Z = 0, reverse-KL q with precision {np.diag(path.prec).round(4)}"
          f"   p: P(v < {NECK}) {p_neck:.4f}   N = {n}, {reps} runs per row")
    print(f"metric: m_x(v) = beta e^-v + (1 - beta) {model.ax[0]:.4f},  m_v = beta {model.gv:.4f} + (1 - beta) {model.av:.4f}")

    print("\n== the step against day 2's at beta = 1, and the roundtrip at every beta: 20 steps of eps 0.2 from 200 draws ==")
    z0 = ais4.draw_p(200)
    fisher = FisherFunnel(D, SIGMA_V)
    p0 = model.draw_p(1.0, z0)
    q, p = z0, p0
    for _ in range(20):
        q, p = model.leap(1.0, q, p, 0.2)
    worst_q = worst_h = 0.0
    for i in range(200):
        x, v, px, pv = list(z0[i, :-1]), z0[i, -1], list(p0[i, :-1]), p0[i, -1]
        h0 = energy(x, v, px, pv, 1.0 / SIGMA_V**2, fisher.gv)
        for _ in range(20):
            x, v, px, pv = explicit_step(x, v, px, pv, 0.2, 1.0 / SIGMA_V**2, fisher.gv)
        worst_q = max(worst_q, np.max(np.abs(np.append(x, v) - q[i])) / (1.0 + np.max(np.abs(q[i]))))
        dh_mine = model.hamiltonian(1.0, q[i:i + 1], p[i:i + 1])[0] - model.hamiltonian(1.0, z0[i:i + 1], p0[i:i + 1])[0]
        worst_h = max(worst_h, abs(energy(x, v, px, pv, 1.0 / SIGMA_V**2, fisher.gv) - h0 - dh_mine))
    print(f"  beta = 1 against day 2's explicit step: state {worst_q:.1e} relative, dH {worst_h:.1e}, worst of 200")
    print("  the starts are draws of p, which is not f_beta's typical set at beta < 1")
    for beta in (0.0, 0.3, 0.7, 1.0):
        for eps in (0.2, 0.4):
            p0 = model.draw_p(beta, z0)
            with np.errstate(all="ignore"):
                q, p = z0, p0
                vmax, vmin = z0[:, -1].copy(), z0[:, -1].copy()
                for _ in range(20):
                    q, p = model.leap(beta, q, p, eps)
                    vmax, vmin = np.maximum(vmax, q[:, -1]), np.minimum(vmin, q[:, -1])
                dh = np.abs(model.hamiltonian(beta, q, p) - model.hamiltonian(beta, z0, p0))
                q, p = q, -p
                for _ in range(20):
                    q, p = model.leap(beta, q, p, eps)
                miss = np.max(np.abs(q - z0), axis=1) / (1.0 + np.max(np.abs(z0), axis=1))
            miss = np.where(np.isfinite(miss), miss, np.inf)
            lost = miss > 1e-6
            print(f"  beta {beta:3.1f}  eps {eps:3.1f}   roundtrip miss median {np.median(miss):.1e}  worst {miss.max():.1e}"
                  f"   lost {int(lost.sum()):3d} of 200   |dH| median {np.nanmedian(dh):.2e}  90th {np.nanpercentile(dh, 90):.2e}"
                  f"  not finite {int(np.sum(~np.isfinite(dh)))}")
            if lost.any():
                v0 = z0[lost, -1]
                print(f"      the lost ones start at v {np.round(np.sort(v0), 1)}"
                      f"   and run to v max {vmax[lost].max():.1f}, min {vmin[lost].min():.1f}"
                      f"   where all 200 start in [{z0[:, -1].min():.1f}, {z0[:, -1].max():.1f}]")

    print(f"\n== smc with the rmhmc move against the hmc move, same seed, cost in gradients per particle ==")
    print("  kernel       rho  moves    T   grads/particle   log Z_hat         sd     P(v<-4.16)       se     ess   min v   div <=0.9   >0.9   ancestors")
    best = {}
    rows_plan = [("hmc 0.2", 0.5, 1), ("hmc 0.2", 0.9, 5), ("hmc 0.2", 0.9, 20), ("hmc 0.2", 0.9, 40),
                 ("rmhmc 0.2", 0.5, 1), ("rmhmc 0.2", 0.9, 1), ("rmhmc 0.2", 0.9, 5), ("rmhmc 0.2", 0.9, 20),
                 ("rmhmc 0.4", 0.9, 5), ("rmhmc 0.4", 0.9, 20), ("rmhmc 0.8", 0.9, 5), ("rmhmc 0.8", 0.9, 20)]
    for kernel, rho, moves in rows_plan:
        rows = [smc4.smc(path, n, rho, moves, kernel) for _ in range(reps)]
        est = np.array([r[0] for r in rows])
        nk = np.array([smc4.neck(r[2]) for r in rows])
        share = nk[:, 0]
        ess = share.mean() * (1 - share.mean()) / max(share.var(ddof=1), 1e-12)
        spent = np.mean([r[5] for r in rows])
        grads = spent * GRADS[kernel]
        best[(kernel, rho, moves)] = (share.mean(), grads, ess, est.mean())
        print(f"  {kernel:10s}  {rho:4.2f}  {moves:4d}  {np.mean([r[1] for r in rows]):5.1f}   {grads:10.0f}"
              f"      {est.mean():+.4f} +- {est.std(ddof=1) / np.sqrt(reps):.4f}  {est.std(ddof=1):.4f}"
              f"    {share.mean():.4f}   {share.std(ddof=1) / np.sqrt(reps):.4f}  {ess:6.0f}   {nk[:, 1].min():6.2f}"
              f"   {np.mean([r[3] for r in rows]):8.0f}  {np.mean([r[4] for r in rows]):6.0f}   {np.mean([r[6] for r in rows]):7.1f}")
    print("  ais at matched moves, same N")
    for T in (100, 300):
        rows = [smc4.ais(path, n, T) for _ in range(reps)]
        est = np.array([r[0] for r in rows])
        nk = np.array([smc4.neck(r[1]) for r in rows])
        share = nk[:, 0]
        ess = share.mean() * (1 - share.mean()) / max(share.var(ddof=1), 1e-12)
        print(f"  ais hmc 0.2   -     1  {T:5d}   {10 * (T - 1):10.0f}      {est.mean():+.4f} +- {est.std(ddof=1) / np.sqrt(reps):.4f}"
              f"  {est.std(ddof=1):.4f}    {share.mean():.4f}   {share.std(ddof=1) / np.sqrt(reps):.4f}  {ess:6.0f}"
              f"   {nk[:, 1].min():6.2f}   {np.mean([r[2] for r in rows]):8.0f}  {np.mean([r[3] for r in rows]):6.0f}")
    print(f"  the ess column is p(1 - p) / Var(p_hat) over the {reps} runs, so it carries about +-50% itself")

    print("\n== the neck, in full: the final cloud's v against 400k draws of p, best row of each kernel ==")
    print("  kernel                   P(v<-4.16)   P(v<-7)   Var(v)   mean v    min v")
    print(f"  p                          {p_neck:.4f}    {np.mean(ref < -7):.4f}   {ref.var():6.2f}   {ref.mean():6.2f}   {ref.min():6.2f}")
    for kernel, rho, moves in (("hmc 0.2", 0.9, 20), ("hmc 0.2", 0.9, 40), ("rmhmc 0.2", 0.9, 20), ("rmhmc 0.4", 0.9, 20)):
        v = np.concatenate([smc4.smc(path, n, rho, moves, kernel)[2][:, -1] for _ in range(4)])
        print(f"  {kernel:10s} rho {rho} x{moves:2d}      {np.mean(v < NECK):.4f}    {np.mean(v < -7):.4f}   {v.var():6.2f}   {v.mean():6.2f}   {v.min():6.2f}")

    print("\n== one run of each kernel at rho = 0.9 and 20 moves, temperature by temperature ==")
    for kernel in ("hmc 0.2", "rmhmc 0.2", "rmhmc 0.8"):
        print(f"  {kernel}")
        smc_trace(path, n, 0.9, 20, kernel)

    print("\n== cost per effective sample of the neck share, gradients per particle x N / ess ==")
    for key, (share, grads, ess, est) in best.items():
        if grads > 0 and ess > 0:
            print(f"  {key[0]:10s} rho {key[1]} x{key[2]:2d}   neck {share:.4f}   grads/particle {grads:7.0f}   ess {ess:6.0f}"
                  f"   gradients per effective sample {grads * n / ess:10.0f}")
    print("  day 2's chain at beta = 1 by hand, eps 0.4, L 20, from its own numbers: 400k gradients for 9000, 44 per effective sample")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
