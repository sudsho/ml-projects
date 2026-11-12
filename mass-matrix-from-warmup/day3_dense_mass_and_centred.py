"""Day 3 - A dense mass from the warmup covariance, and the centred model.

Day 2 left log tau failing under the diagonal mass on the non-centred model,
0.47 alone, and the project's question about the centred one, whose scale
changes with position, was not touched. This file estimates the whole
covariance in day 1's windows and not only its diagonal, each chain its own
matrix, and runs day 2's tree under it: the momentum N(0, M), the position
along minv p, the U-turn on velocities. Four runs on the same 40 studies of
500, non-centred or centred, diagonal or dense, each from its own warmup.
Written before the run: the dense mass does nothing for log tau on the
non-centred model, about 0.5 again, because nothing there is correlated; on
the centred model the diagonal mass shortens the tree from the unit-mass 148
steps, divergences stay between 1 and 2% and the band at 100%; and the dense
mass is no better than the diagonal one there.

0. The tree is right under a matrix. On a correlated ten-dimensional normal
   the variances are 0.994 to 1.006 of the truth under unit, diagonal and
   dense mass and no correlation is off by more than 0.006. The exact dense
   mass stops at depth 4.25, which is day 2's exact diagonal mass on an
   uncorrelated normal to the digit.

1. Non-centred, nothing to find. The median estimated correlation is 0.00
   on three of the four pairs printed and -0.05 on mu with eta_1, and the
   correlation matrix still has condition number 10.6, 19.3 at the 90th
   percentile: a hundred correlated draws in ten dimensions make that out
   of noise. The tree gets longer, 55.4 steps against 40.3, depth 6 on 42%
   against 12%. log tau alone is 0.38 against 0.28 and the four together
   0.50 against 0.42, lag-1 +0.38 against +0.42. Right: no better, and
   37% more gradients for it.

2. Day 2's 0.60 was partly the draw. The same sampler on these studies is
   0.42 on the four and 0.28 on log tau, 40.3 steps both days. Still a
   fail and still log tau, with the same 1.06 in its lowest tenth, but 40
   studies move the share by 0.2 and day 4 should not read less than that.

3. Centred, diagonal. The tree is 61.5 steps against 148 under unit mass,
   depth 5 and 6 on two thirds, so the mass does shorten it, by putting
   theta_1 at 11.5 and log tau at 0.29. Divergences are 1.08% of
   transitions on 19% of chains, the band 100% from 100 simulations, and
   log tau's lag-1 is +0.80. Its histogram is 1.94 in the lowest tenth
   and 1.44 in the highest. Right on all three.

4. Centred, dense. Here there is something to find, mu with theta_1 +0.57
   and theta_1 with theta_2 +0.37, and the matrix that finds it has
   condition number 176 at the median and 1257 at the 90th percentile.
   Divergences fall five times, 0.20% on 7% of chains, which I did not
   predict, and mu's lag-1 goes +0.50 to +0.23. But the tree is 149 steps
   again, depth 7 or 8 on 57%, log tau's lag-1 is +0.80 exactly as before
   and the band is 100% from 100 simulations with 1.79 and 1.51 at the
   ends. Half: no better on the verdict, better on the divergences, and
   2.4 times the gradients.

5. Where the centred failure sits. On studies whose true tau is below 1,
   log tau's rank is in the lowest tenth on 0.83 of simulations under the
   diagonal mass and 0.78 under the dense, against 0.62 on the non-centred
   runs, which is what a calibrated sampler gives there. From 1 to 5 it is
   0.12 and 0.11 against 0.02. Divergences do not sort by tau, 1.14% /
   1.16% / 0.91%. The chain is too high on log tau wherever the truth is
   low, which is the neck not entered, and no constant matrix has the neck
   in it.

6. Cost. 8498 gradients a simulation with warmup for the non-centred
   diagonal fail, 10748 for the dense one, 11662 and 24711 for the two
   centred ones. Three right and one half, and the first day of this
   project where the prediction mostly held. Day 4 is what the warmup
   costs, on the non-centred diagonal mass, the only one near a pass.

NumPy only. Fixed seeds. About 2900 seconds, 200 of them the check.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_diagonal_mass_sweep as m1  # noqa: E402  (the windows; it puts the older projects on the path)
from day1_diagonal_mass_sweep import d1, d2, es, ends, per_param, shape, KEYS, PARAMS, draw_studies, params  # noqa: E402

# Day 1 reseeds on import; reseed again so these studies are fresh draws.
es.RNG = np.random.default_rng(141)
d1.RNG = np.random.default_rng(142)
d2.RNG = np.random.default_rng(143)
RNG = np.random.default_rng(144)

L, SIZES = d1.L, d1.SIZES
EPS = m1.EPS
STUDIES, PASS, LAG_CHAINS = m1.STUDIES, m1.PASS, m1.LAG_CHAINS
BURN = m1.BURN                   # transitions of the run's own sampler from the warmed state
MAX_DEPTH = 8                    # 255 leapfrog steps, as on days 2 and the unit-mass NUTS
WARM_COST = (m1.FAST + sum(m1.WINDOWS)) * m1.WARM_STEPS
D = 10
RUNS = [("noncentred", "diagonal"), ("noncentred", "dense"), ("centred", "diagonal"), ("centred", "dense")]
# the pairs whose estimated correlation is printed: columns are mu, log tau, then the eight etas or thetas
PAIRS = [("mu, log tau", 0, 1), ("mu, fine_1", 0, 2), ("log tau, fine_1", 1, 2), ("fine_1, fine_2", 2, 3)]


def vel(minv, p):
    """Velocity minv p, one matrix a chain; minv is (c, d, d)."""
    return np.matmul(minv, p[:, :, None])[:, :, 0]


def mass_of(cov):
    """A mass as the sampler takes it: the inverse mass (c, d, d) and the map from white noise to momentum.

    Momentum is N(0, M) with M the inverse of cov. With cov = C C' from
    Cholesky, p = C^-T g has covariance C^-T C^-1 = M, so the second array is
    the transpose of C's inverse, computed once a chain and not once a draw.
    """
    chol = np.linalg.cholesky(cov)
    return cov, np.linalg.inv(chol).transpose(0, 2, 1)


def unit_mass(c, d=D):
    return mass_of(np.tile(np.eye(d), (c, 1, 1)))


def regularise(cov, n, dense):
    """Stan's shrinkage of a window's estimate toward 1e-3 times the identity, weight 5 / (n + 5).

    The diagonal form keeps the variances and drops everything else before
    shrinking, so it is day 1's mass written as a matrix.
    """
    if not dense:
        cov = cov * np.eye(cov.shape[1])
    return (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * np.eye(cov.shape[1])


def hmc_dense(z, steps, mass, pot_grad):
    """Fixed-length HMC under a per-chain dense mass; day 1's hmc_mass with minv a matrix."""
    pot, grad = pot_grad
    minv, a = mass
    p = vel(a, es.RNG.standard_normal(z.shape))
    h0 = pot(z) + 0.5 * (p * vel(minv, p)).sum(1)
    zn = z.copy()
    g = grad(zn)
    for _ in range(steps):
        p = p - 0.5 * EPS * g
        zn = zn + EPS * vel(minv, p)
        g = grad(zn)
        p = p - 0.5 * EPS * g
    dh = pot(zn) + 0.5 * (p * vel(minv, p)).sum(1) - h0
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(es.RNG.random(z.shape[0])) < -np.nan_to_num(dh, nan=np.inf))
    return np.where(acc[:, None], zn, z), acc


def warmup(r, kind, dense):
    """Day 1's windows on either model, the estimate a covariance matrix or its diagonal.

    A fast window at unit mass, then slow windows of 25, 50 and 100, every
    transition one unit long, the mass re-estimated from each window alone.
    Returns the state, the final mass, the final covariance estimate and the
    acceptance over the last window.
    """
    pg = es.MODELS[kind]
    z = es.prior_start(kind, r)
    mass = unit_mass(r)
    for _ in range(m1.FAST):
        z, _ = hmc_dense(z, m1.WARM_STEPS, mass, pg)
    acc = 0.0
    for n in m1.WINDOWS:
        s1 = np.zeros((r, D))
        s2 = np.zeros((r, D, D))
        acc = 0.0
        for _ in range(n):
            z, a = hmc_dense(z, m1.WARM_STEPS, mass, pg)
            acc += a.mean() / n
            s1 += z
            s2 += z[:, :, None] * z[:, None, :]
        m = s1 / n
        cov = regularise(s2 / n - m[:, :, None] * m[:, None, :], n, dense)
        mass = mass_of(cov)
    return z, mass, cov, acc


def nuts_dense(z, eps, mass, pot_grad, max_depth=MAX_DEPTH):
    """One NUTS transition on every chain at once under a per-chain dense mass.

    Day 2's tree with the multinomial draw. The mass enters in three places:
    the momentum is N(0, M), the position moves along the velocity minv p,
    and the U-turn compares the summed momentum with the velocities at the
    two ends of each block and of the whole tree. With minv diagonal it is
    day 2's sampler, and with the identity the trajectory length project's.

    Returns (z, depth reached, leapfrog steps taken, diverged).
    """
    pot, grad = pot_grad
    minv, a = mass
    c, d = z.shape
    p0 = vel(a, es.RNG.standard_normal(z.shape))
    v0 = vel(minv, p0)
    g0 = grad(z)
    h0 = pot(z) + 0.5 * (p0 * v0).sum(1)
    zm, pm, gm, vm = z.copy(), p0.copy(), g0.copy(), v0.copy()     # the end earliest in time
    zp, pp, gp, vp = z.copy(), p0.copy(), g0.copy(), v0.copy()     # the end latest in time
    prop, lw, rho = z.copy(), np.zeros(c), p0.copy()
    live = np.ones(c, bool)
    depth = np.zeros(c, np.int32)
    steps = np.zeros(c, np.int32)
    diverged = np.zeros(c, bool)
    for j in range(max_depth):
        fwd = RNG.random(c) < 0.5
        f = fwd[:, None]
        zs, ps, gs = np.where(f, zp, zm), np.where(f, pp, pm), np.where(f, gp, gm)
        vs = np.where(f, vp, vm)
        e = np.where(fwd, eps, -eps)[:, None]
        ok = live.copy()
        s_lw = np.full(c, -np.inf)
        s_prop = zs.copy()
        S = np.zeros((c, d))
        ck_v = np.empty((j + 1, c, d))
        ck_S = np.empty((j + 1, c, d))
        for n in range(2 ** j):
            p1 = ps - 0.5 * e * gs
            z1 = zs + e * vel(minv, p1)
            g1 = grad(z1)
            p1 = p1 - 0.5 * e * g1
            v1 = vel(minv, p1)
            step = ok[:, None]
            zs, ps, gs = np.where(step, z1, zs), np.where(step, p1, ps), np.where(step, g1, gs)
            vs = np.where(step, v1, vs)
            steps += ok
            dh = pot(zs) + 0.5 * (ps * vs).sum(1) - h0
            bad = ok & (~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000))
            diverged |= bad
            ok &= ~bad
            leaf = np.where(np.isfinite(dh), -dh, -np.inf)
            grown = np.logaddexp(s_lw, leaf)
            take = ok & (np.log(RNG.random(c)) < leaf - grown)
            s_prop = np.where(take[:, None], zs, s_prop)
            s_lw = np.where(ok, grown, s_lw)
            for k in range(1, j + 1):
                if n % 2 ** k == 0:              # this leaf opens a block of 2^k
                    ck_v[k], ck_S[k] = vs, S
            S = S + ps
            for k in range(1, j + 1):
                if (n + 1) % 2 ** k == 0:        # this leaf closes one
                    r = S - ck_S[k]
                    ok &= ((r * ck_v[k]).sum(1) > 0) & ((r * vs).sum(1) > 0)
        merge = ok
        mf, mb = (merge & fwd)[:, None], (merge & ~fwd)[:, None]
        zp, pp, gp, vp = np.where(mf, zs, zp), np.where(mf, ps, pp), np.where(mf, gs, gp), np.where(mf, vs, vp)
        zm, pm, gm, vm = np.where(mb, zs, zm), np.where(mb, ps, pm), np.where(mb, gs, gm), np.where(mb, vs, vm)
        take = merge & (np.log(RNG.random(c)) < s_lw - lw)
        prop = np.where(take[:, None], s_prop, prop)
        lw = np.where(merge, np.logaddexp(lw, s_lw), lw)
        rho = np.where(merge[:, None], rho + S, rho)
        depth = np.where(merge, j + 1, depth)
        live = merge & ((rho * vm).sum(1) > 0) & ((rho * vp).sum(1) > 0)
        if not live.any():
            break
    return prop, depth, steps, diverged


def check_on_normal(form, chains=2000, n=150, burn=50, eps=0.2):
    """A correlated ten-dimensional normal, scales 0.5 to 5, every pair correlated 0.9 before a rotation.

    form = "unit", "diagonal" (the true variances) or "dense" (the true
    covariance). Returns variance over truth, the worst error of a
    correlation, and the mean depth, which is what the form is for.
    """
    s = np.linspace(0.5, 5.0, D)
    corr = np.full((D, D), 0.9) + 0.1 * np.eye(D)
    q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((D, D)))
    corr = q @ np.diag(np.linalg.eigvalsh(corr)) @ q.T
    sd = np.sqrt(np.diag(corr))
    corr = corr / np.outer(sd, sd)
    cov = corr * np.outer(s, s)
    prec = np.linalg.inv(cov)
    pg = (lambda z: 0.5 * ((z @ prec) * z).sum(1), lambda z: z @ prec)
    minv = {"unit": np.eye(D), "diagonal": np.diag(s**2), "dense": cov}[form]
    mass = mass_of(np.tile(minv, (chains, 1, 1)))
    z = es.RNG.standard_normal((chains, D)) @ np.linalg.cholesky(cov).T
    sm1 = np.zeros(D)
    sm2 = np.zeros((D, D))
    dep = 0.0
    for t in range(burn + n):
        z, dp, _, _ = nuts_dense(z, eps, mass, pg)
        if t >= burn:
            sm1 += z.mean(0)
            sm2 += z.T @ z / chains
            dep += dp.mean()
    c = sm2 / n - np.outer(sm1 / n, sm1 / n)
    var = np.diag(c)
    return var / s**2, np.abs(c / np.sqrt(np.outer(var, var)) - corr).max(), dep / n


def sbc_nuts(z, truth, mass, kind):
    """BURN transitions from the warmed state, then L with the ranks, pilots, depth, steps and divergences."""
    pg = es.MODELS[kind]
    r = z.shape[0]
    for _ in range(BURN):
        z, _, _, _ = nuts_dense(z, EPS, mass, pg)
    below = np.zeros((r, 4), dtype=np.int32)
    kept = np.empty((L, LAG_CHAINS, 3))
    depth = np.zeros(MAX_DEPTH + 1)
    steps = np.zeros(r)
    div = np.zeros(r)
    moved = 0.0
    for k in range(L):
        zn, dp, st, dv = nuts_dense(z, EPS, mass, pg)
        moved += (zn != z).any(1).mean()
        z = zn
        depth += np.bincount(dp, minlength=MAX_DEPTH + 1)
        steps += st
        div += dv
        p = params(kind, z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, kept, depth / depth.sum(), steps / L, div / L, moved / L


def describe(cov):
    """Median inverse mass of mu, log tau and the first fine coordinate, the printed correlations, the condition number."""
    sd = np.sqrt(np.einsum("cii->ci", cov))
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    cond = np.linalg.cond(corr)
    return ("minv " + " ".join(f"{np.median(sd[:, j] ** 2):6.2f}" for j in (0, 1, 2))
            + "   corr " + "  ".join(f"{name} {np.median(corr[:, i, j]):+.2f}" for name, i, j in PAIRS)
            + f"   cond {np.median(cond):.1f} ({np.quantile(cond, 0.9):.1f} at 90%)")


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    print("0. correlated ten-dimensional normal, eps 0.2, 2000 chains x 150")
    for form in ("unit", "diagonal", "dense"):
        vr, ce, dep = check_on_normal(form)
        print(f"  {form:8s} mass  variance / truth {vr.min():.3f} to {vr.max():.3f}  worst correlation error {ce:.3f}  "
              f"mean depth {dep:.2f}  {time.time() - t0:.0f} s", flush=True)

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    truth, y = draw_studies(r)
    es.Y = y
    print(f"\nmass from warmup, day 3: nuts at eps {EPS}, max depth {MAX_DEPTH}, {STUDIES} x {nmax} studies, "
          f"L = {L}, day 1's windows ({WARM_COST} gradients) on each model and each form, burn {BURN}")

    runs = {}
    print("\n1. the estimate after warmup, medians over chains: inverse mass of mu, log tau, fine_1 "
          "(eta_1 or theta_1), four correlations, the condition number of the correlation matrix")
    print("2. steps a transition (mean, median, 90th), depth shares 3-8, divergences, share of transitions "
          "that moved, lag-1 of mu / log tau / eta_1")
    for kind, form in RUNS:
        z0, mass, cov, wacc = warmup(r, kind, form == "dense")
        print(f"  {kind:10s} {form:8s}  1. {describe(cov)}   last window acc {wacc:.3f}  {time.time() - t0:5.0f} s",
              flush=True)
        below, kept, dshare, steps, div, moved = sbc_nuts(z0, truth, mass, kind)
        runs[kind, form] = (below.reshape(STUDIES, nmax, 4), steps, div)
        rho = m1.lag1(kept)
        print(f"  {kind:10s} {form:8s}  2. steps {steps.mean():5.1f} {np.median(steps):5.1f} "
              f"{np.quantile(steps, 0.9):5.1f}  depth " + " ".join(f"{x:.2f}" for x in dshare[3:])
              + f"  div {100 * div.mean():.2f}% on {(div > 0).mean():.3f} of chains  moved {moved:.3f}  "
              f"rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f}  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n3. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for (kind, form), (rr, _, _) in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  {kind:10s} {form:8s}  {cells}")

    print(f"\n4. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for (kind, form), (rr, _, _) in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  {kind:10s} {form:8s}  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n5. pooled rank histograms of mu, log tau and theta_1 in tenths at n = {nmax}")
    for (kind, form), (rr, _, _) in runs.items():
        print(f"  {kind:10s} {form:8s}  mu {shape(rr, 0)}   log tau {shape(rr, 1)}   theta_1 {shape(rr, 2)}")

    print("\n6. by the study's true tau, below 1 / 1-5 / above 5: mean steps, divergences a transition, "
          "and the share of log tau's ranks in the lowest tenth")
    tau = np.exp(truth[:, 1])
    bins = [tau < 1, (tau >= 1) & (tau < 5), tau >= 5]
    for (kind, form), (rr, steps, div) in runs.items():
        low = (rr.reshape(r, 4)[:, 1] < 0.1 * (L + 1))
        print(f"  {kind:10s} {form:8s}  steps " + " / ".join(f"{steps[b].mean():5.1f}" for b in bins)
              + "   div " + " / ".join(f"{100 * div[b].mean():.2f}%" for b in bins)
              + "   lowest tenth " + " / ".join(f"{low[b].mean():.3f}" for b in bins))

    print(f"\n7. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}, gradients a "
          f"simulation = mean steps x ({BURN} + {L}) plus the warmup's {WARM_COST}")
    for (kind, form), (rr, steps, _) in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        cost = steps.mean() * (BURN + L) + WARM_COST
        print(f"  {kind:10s} {form:8s}  band {share:.2f}  {'pass' if share <= PASS else 'fail'}  "
              f"worst {worst:8s} {steps.mean() * L:6.0f} sampling, {cost:6.0f} in all")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
