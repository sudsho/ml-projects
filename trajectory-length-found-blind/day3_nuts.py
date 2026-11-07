"""Day 3 - NUTS from scratch: the doubling tree, the U-turn and the multinomial draw.

Days 1 and 2 found the trajectory length by sweeping it. Fixed lengths at
eps 0.1 passed at four units and fourteen and nowhere else, and jitter turned
four to ten units into a plateau, with mu at three units still failing for
being too short (rho1 +0.58). NUTS is meant to find the length without the
sweep: double the trajectory in a random direction until its ends start
coming back towards each other, then draw the next state from the whole
tree in proportion to its weight. This file writes it from scratch on every
chain at once (unit mass, max depth 8, the generalised criterion checked on
every aligned block, biased progressive sampling between subtrees), checks
it on a ten-dimensional normal, and runs it through day 1's SBC with data on
non-centred at eps 0.1 and 0.2 and centred at eps 0.1. No prediction was
written before the run; the project's question was whether the U-turn lands
in the window the sweep found.

0. The sampler is right. On a normal with scales 0.5 to 5 the variances are
   0.995 to 1.004 of the truth and the means within 0.003 scales, 4000
   chains x 200, mean depth 5.7.

1. The length lands in the window and the chain still fails. Non-centred
   at eps 0.1 stops at 41.6 steps a transition, 4.2 units, median 41.7 and
   90th 44.6, depth 5 on 88% of transitions. At eps 0.2 it takes 22.2 steps
   and 4.4 units. So the length is set by the posterior and not the step
   size, and it is day 1's four-unit window to within 0.4 units, found with
   no sweep. But the band at 500 rejects 97% of studies at both step sizes,
   against 0.20 for fixed four units on day 1, and the parameter is mu:
   band 0.90 and 0.95 alone, a U with 1.19 and 1.16 in the end tenths, lag-1
   +0.74 and +0.71. Log tau is at 0.35 and 0.30, theta_1 at 0.12, eta_1 at
   0.05 and 0.07 with a flat histogram.

2. Why, as far as this run goes. The U-turn sum on unit mass is over all
   ten coordinates and eight of them are the etas, near unit normal, which
   turn at about pi units; mu's clock is near twenty (day 1). So the tree
   stops when the etas turn, at a length that is fine for the etas and
   short for mu, which is day 2's mu-at-three-units failure again (+0.58
   there, +0.74 here). The multinomial draw then takes a state from inside
   the tree rather than its end, so mu moves less than four units of
   integration would move it. That last part is the likely reason the lag
   is worse than fixed four units, and it is not measured here; drawing
   from the tree's far end instead would test it.

3. The rank mean still passes. Mean rejects 0.20
   and 0.05 of studies at 500 on non-centred, 0.17 on centred, while the
   band and variance tests catch the U. Same as every project since the
   rank tests one: a symmetric U leaves the mean where it should be.

4. Centred is a failure of a different kind. The tree goes deep, depth 7
   on 42% and the cap of 8 on 23% of transitions, 147.9 steps mean and 14.8
   units, with divergences on 1.87% of transitions, about the same in every
   tau bin (1.84%, 1.93%, 1.78%), and it is rejected on 100% of studies by
   n = 100, log tau alone at 1.00 with lag-1 +0.80. NUTS spends its length
   on the funnel and does not get out of it.

5. Cost. Non-centred is 4121 gradients a simulation at eps 0.1 and 2194 at
   eps 0.2, against 3960 for the fixed four units that passed on day 1, so
   the price is right and the chain is not. Centred is 14645 for nothing.
   The length barely moves with the study's tau, 41.8 / 41.8 / 41.3 steps
   below 1, 1 to 5 and above 5 on non-centred, which says again that the
   etas decide it.

The result for a blind rule: NUTS finds the window's length and fails in it,
because what it measures is when the many unit-scale coordinates turn, and
the one wide coordinate needs longer. Day 4 sets the length from the IACT of
the widest parameter during warmup, which is the opposite choice, and should
be checked against this file's number for mu.

NumPy only. Fixed seeds. About 1690 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_period_sweep as d0  # noqa: E402  (puts the eight schools and rank-test modules on the path)
from day1_period_sweep import d1, d2, es, ends, per_param, shape, KEYS, PARAMS, draw_studies, params  # noqa: E402

# Day 1 reseeds on import; reseed again so the NUTS studies are fresh draws.
es.RNG = np.random.default_rng(101)
d1.RNG = np.random.default_rng(102)
d2.RNG = np.random.default_rng(103)
RNG = np.random.default_rng(104)

L, SIZES = d1.L, d1.SIZES
BURN, STUDIES, PASS = d0.BURN, d0.STUDIES, d0.PASS
LAG_CHAINS = d0.LAG_CHAINS
MAX_DEPTH = 8                    # 255 leapfrog steps, 25.5 units at eps 0.1
RUNS = [("noncentred", 0.1), ("noncentred", 0.2), ("centred", 0.1)]


def nuts(kind, z, eps, max_depth=MAX_DEPTH, pot_grad=None):
    """One multinomial NUTS transition on every chain at once, unit mass.

    Each doubling picks a direction per chain and builds a new subtree of 2^j
    leapfrog steps off that end of the tree. Inside the subtree the proposal is
    drawn by progressive multinomial sampling (each new leaf replaces it with
    its share of the subtree's weight), and every aligned block of 2, 4, ...
    leaves is checked for a U-turn as soon as its last leaf exists, using the
    block's end momenta and the sum of its momenta (the generalised criterion,
    which on unit mass is the sum itself). A subtree that turns or diverges is
    thrown away and the chain stops; one that completes replaces the tree's
    proposal with probability min(1, w_new / w_old), the biased progressive
    step, and the whole tree is then checked. All chains step in lockstep and a
    stopped chain is frozen with np.where, so no chain's tree depends on another's.

    Returns (z, depth reached, leapfrog steps taken, diverged).
    """
    pot, grad = pot_grad if pot_grad is not None else es.MODELS[kind]
    c, d = z.shape
    p0 = es.RNG.standard_normal(z.shape)
    g0 = grad(z)
    h0 = pot(z) + 0.5 * (p0 * p0).sum(1)
    zm, pm, gm = z.copy(), p0.copy(), g0.copy()     # the end earliest in time
    zp, pp, gp = z.copy(), p0.copy(), g0.copy()     # the end latest in time
    prop, lw, rho = z.copy(), np.zeros(c), p0.copy()
    live = np.ones(c, bool)
    depth = np.zeros(c, np.int32)
    steps = np.zeros(c, np.int32)
    diverged = np.zeros(c, bool)
    for j in range(max_depth):
        fwd = RNG.random(c) < 0.5
        f = fwd[:, None]
        zs, ps, gs = np.where(f, zp, zm), np.where(f, pp, pm), np.where(f, gp, gm)
        e = np.where(fwd, eps, -eps)[:, None]
        ok = live.copy()
        s_lw = np.full(c, -np.inf)
        s_prop = zs.copy()
        S = np.zeros((c, d))
        ck_p = np.empty((j + 1, c, d))
        ck_S = np.empty((j + 1, c, d))
        for n in range(2 ** j):
            p1 = ps - 0.5 * e * gs
            z1 = zs + e * p1
            g1 = grad(z1)
            p1 = p1 - 0.5 * e * g1
            step = ok[:, None]
            zs, ps, gs = np.where(step, z1, zs), np.where(step, p1, ps), np.where(step, g1, gs)
            steps += ok
            dh = pot(zs) + 0.5 * (ps * ps).sum(1) - h0
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
                    ck_p[k], ck_S[k] = ps, S
            S = S + ps
            for k in range(1, j + 1):
                if (n + 1) % 2 ** k == 0:        # this leaf closes one
                    r = S - ck_S[k]
                    ok &= ((r * ck_p[k]).sum(1) > 0) & ((r * ps).sum(1) > 0)
        merge = ok
        mf, mb = (merge & fwd)[:, None], (merge & ~fwd)[:, None]
        zp, pp, gp = np.where(mf, zs, zp), np.where(mf, ps, pp), np.where(mf, gs, gp)
        zm, pm, gm = np.where(mb, zs, zm), np.where(mb, ps, pm), np.where(mb, gs, gm)
        take = merge & (np.log(RNG.random(c)) < s_lw - lw)
        prop = np.where(take[:, None], s_prop, prop)
        lw = np.where(merge, np.logaddexp(lw, s_lw), lw)
        rho = np.where(merge[:, None], rho + S, rho)
        depth = np.where(merge, j + 1, depth)
        live = merge & ((rho * pm).sum(1) > 0) & ((rho * pp).sum(1) > 0)
        if not live.any():
            break
    return prop, depth, steps, diverged


def check_on_normal(chains=4000, n=200, burn=50, eps=0.2):
    """NUTS on a ten-dimensional normal with scales 0.5 to 5: variance ratios and mean depth."""
    s = np.linspace(0.5, 5.0, 10)
    pg = (lambda z: 0.5 * ((z / s) ** 2).sum(1), lambda z: z / s**2)
    z = s * es.RNG.standard_normal((chains, 10))
    m1 = np.zeros(10)
    m2 = np.zeros(10)
    dep = 0.0
    for t in range(burn + n):
        z, dp, _, _ = nuts(None, z, eps, pot_grad=pg)
        if t >= burn:
            m1 += z.mean(0)
            m2 += (z * z).mean(0)
            dep += dp.mean()
    var = m2 / n - (m1 / n) ** 2
    return var / s**2, m1 / n / s, dep / n


def sbc_pilot(truth, y, kind, eps):
    """Day 1's pilot with the NUTS transition: ranks, pilots for lag-1, and per-transition depth, steps, divergences."""
    es.Y = y
    r = y.shape[0]
    z = es.prior_start(kind, r)
    for _ in range(BURN):
        z, _, _, _ = nuts(kind, z, eps)
    below = np.zeros((r, 4), dtype=np.int32)
    kept = np.empty((L, LAG_CHAINS, 3))
    depth = np.zeros(MAX_DEPTH + 1)
    steps = np.zeros(r)
    div = np.zeros(r)
    for k in range(L):
        z, dp, st, dv = nuts(kind, z, eps)
        depth += np.bincount(dp, minlength=MAX_DEPTH + 1)
        steps += st
        div += dv
        p = params(kind, z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, kept, depth / depth.sum(), steps / L, div / L


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    vr, mz, dep = check_on_normal()
    print("0. ten-dimensional normal, scales 0.5 to 5, eps 0.2, 4000 chains x 200")
    print(f"  variance / truth {vr.min():.3f} to {vr.max():.3f}  mean / scale within {np.abs(mz).max():.3f}  "
          f"mean depth {dep:.2f}  {time.time() - t0:.0f} s", flush=True)

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    truth, y = draw_studies(r)
    print(f"\ntrajectory length, nuts: max depth {MAX_DEPTH}, {STUDIES} x {nmax} studies, burn {BURN}, L = {L}")

    runs = {}
    print("\n1. steps a transition (mean, median, 90th), length in units, depth shares 3-8, divergences, "
          "lag-1 of mu / log tau / eta_1")
    for kind, eps in RUNS:
        below, kept, dshare, steps, div = sbc_pilot(truth, y, kind, eps)
        runs[kind, eps] = (below.reshape(STUDIES, nmax, 4), steps, div)
        rho = d0.lag1(kept)
        print(f"  {kind:10s} eps {eps}  steps {steps.mean():5.1f} {np.median(steps):5.1f} {np.quantile(steps, 0.9):5.1f}  "
              f"T {eps * steps.mean():4.1f}  depth " + " ".join(f"{x:.2f}" for x in dshare[3:])
              + f"  div {100 * div.mean():.2f}%  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f}  "
              f"{time.time() - t0:5.0f} s", flush=True)

    print(f"\n2. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for (kind, eps), (rr, _, _) in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  {kind:10s} eps {eps}  {cells}")

    print(f"\n3. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for (kind, eps), (rr, _, _) in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  {kind:10s} eps {eps}  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n4. pooled rank histograms of eta_1 and mu in tenths at n = {nmax}")
    for (kind, eps), (rr, _, _) in runs.items():
        print(f"  {kind:10s} eps {eps}  eta_1 {shape(rr, 3)}   mu {shape(rr, 0)}")

    print("\n5. where the length goes: mean steps by the study's true tau, below 1 / 1-5 / above 5")
    tau = np.exp(truth[:, 1])
    bins = [tau < 1, (tau >= 1) & (tau < 5), tau >= 5]
    for (kind, eps), (_, steps, div) in runs.items():
        print(f"  {kind:10s} eps {eps}  steps " + " / ".join(f"{steps[b].mean():5.1f}" for b in bins)
              + "  div " + " / ".join(f"{100 * div[b].mean():.2f}%" for b in bins))

    print(f"\n6. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}, "
          f"gradients a simulation = mean steps x {L}")
    for (kind, eps), (rr, steps, _) in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        print(f"  {kind:10s} eps {eps}  band {share:.2f}  {'pass' if share <= PASS else 'fail'}  "
              f"worst {worst:8s} {steps.mean() * L:7.0f} gradients")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
