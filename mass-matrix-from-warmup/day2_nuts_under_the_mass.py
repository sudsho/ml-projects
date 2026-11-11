"""Day 2 - NUTS under the adapted mass, and the draw from the tree's end.

The trajectory length project's NUTS stopped at four units and failed on mu,
band 0.90 at 500 with lag-1 +0.74, and left a guess it did not test: that the
multinomial draw takes a state from inside the tree, so mu moves less than
the tree is long, and a draw from the tree's far end would show it. Day 1
here built a diagonal mass from warmup that gave mu and the etas one clock
for half a turn. This file runs that project's tree under a per-chain mass
(the U-turn on velocities) with either draw, four runs from the same warmed
state on the same 40 studies of 500: unit or adapted mass, multinomial or
end draw. Written before the run: under the adapted mass the tree stops near
pi units, mu's band drops from 0.90 to a pass and log tau becomes the worst
parameter; under unit mass the end draw brings mu's lag-1 from +0.74 towards
zero and the band on mu under 0.5.

0. The end draw is not a sampler under unit mass. On the ten-dimensional
   normal the multinomial draw gives variances 0.994 to 1.004 of the truth
   under either mass. The end draw gives 0.998 to 1.003 when the mass is
   exact, where every start builds the same tree (mean depth 4.25), and
   0.952 to 1.057 under unit mass, where the depth depends on where the
   start sits. So its ranks under unit mass are a biased chain's.

1. Unit mass, multinomial, is day 3 again on fresh studies: 41.6 steps,
   depth 5 on 88%, mu's lag-1 +0.74, band on mu 0.93, a U with 1.19 in both
   end tenths.

2. The guess is wrong. With the end draw mu's lag-1 is +0.77, not lower,
   with 3.5% of the end states rejected on their energy. The band on mu does
   fall, 0.93 to 0.40, which is the number predicted, but mu's histogram has
   turned over, 0.88 in the end tenths and 1.13 in the middle: the bias of
   item 0 makes the draws too wide and that cancels the U of a slow chain.
   log tau gets worse, 0.38 to 0.55. The draw inside the tree was never what
   held mu back; the tree is short for mu wherever in it the state is taken.

3. The mass fixes mu and does not change the length. Under the adapted mass
   the tree is depth 5 on 86%, 40.3 steps against 41.6, so "stops near pi"
   was true before the mass as well: 31 steps of tree and the rest spent on
   the doubling that turned. What changes is that 3.1 units is now half a
   turn for mu too: lag-1 +0.08, band on mu alone 0.03, histogram flat to
   0.03. Right on mu, and the length prediction was not a prediction.

4. log tau is what fails now. Band 0.47 alone and 0.60 on the four, lag-1
   +0.40 against +0.34 under unit mass, and the shape is not a U: 1.12 in
   the lowest tenth and 0.92 next to it, the rank mean passing at 0.05. Its
   inverse mass is 0.73 with chains from 0.33 to 1.67, the widest spread of
   any coordinate, and day 1 found it has no clock at any length. Right
   that it would be the worst; a diagonal mass does nothing for it.

5. The length does not move with the study's tau under either mass, 40.4 /
   40.3 / 40.2 steps below 1, 1 to 5 and above 5.

6. Cost. Adapted multinomial is 3987 gradients of sampling a simulation
   against 4120 under unit mass, and 8501 with the burn and the 2500 of
   warmup, for 0.60 where day 1's fixed two units passed at 5480. Three
   right, one of them for the wrong reason, one wrong. Day 3 takes the
   dense mass to log tau and the centred model.

NumPy only. Fixed seeds. About 2100 seconds, 400 of them the checks.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_diagonal_mass_sweep as m1  # noqa: E402  (the warmup; it puts the older projects on the path)
from day1_diagonal_mass_sweep import d0, d1, d2, es, ends, per_param, shape, KEYS, PARAMS, draw_studies, params  # noqa: E402

# Day 1 reseeds on import; reseed again so these studies are fresh draws.
es.RNG = np.random.default_rng(131)
d1.RNG = np.random.default_rng(132)
d2.RNG = np.random.default_rng(133)
RNG = np.random.default_rng(134)

L, SIZES = d1.L, d1.SIZES
EPS = m1.EPS
STUDIES, PASS, LAG_CHAINS = m1.STUDIES, m1.PASS, m1.LAG_CHAINS
BURN = m1.BURN                   # transitions of the run's own sampler from the warmed state
MAX_DEPTH = 8                    # 255 leapfrog steps, as in the unit-mass NUTS
WARM_COST = (m1.FAST + sum(m1.WINDOWS)) * m1.WARM_STEPS
RUNS = [("unit", "multinomial"), ("unit", "end"), ("adapted", "multinomial"), ("adapted", "end")]


def nuts_mass(z, eps, minv, draw="multinomial", max_depth=MAX_DEPTH, pot_grad=None):
    """One NUTS transition on every chain at once under a per-chain diagonal mass.

    The tree is the trajectory length project's: a doubling in a random
    direction per chain, progressive multinomial sampling inside the new
    subtree, the generalised U-turn check on every aligned block and on the
    whole tree. What the mass changes is the momentum, drawn N(0, M), the
    kinetic energy 0.5 sum minv p^2, the position step eps minv p, and the
    U-turn, which compares the summed momentum with the end velocities
    minv p and not the end momenta. With minv all ones it is that project's
    sampler line for line.

    draw = "multinomial" returns the state drawn from the whole tree by weight.
    draw = "end" throws that draw away and returns one of the tree's two ends,
    a coin each, accepted on its own energy error. That is a correct kernel
    when every start in a tree would have built the same tree at the same
    depth and not otherwise, so check_on_normal runs it too.

    Returns (z, depth reached, leapfrog steps taken, diverged).
    """
    pot, grad = pot_grad if pot_grad is not None else es.MODELS["noncentred"]
    c, d = z.shape
    p0 = es.RNG.standard_normal(z.shape) / np.sqrt(minv)
    g0 = grad(z)
    h0 = pot(z) + 0.5 * (minv * p0 * p0).sum(1)
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
            z1 = zs + e * minv * p1
            g1 = grad(z1)
            p1 = p1 - 0.5 * e * g1
            step = ok[:, None]
            zs, ps, gs = np.where(step, z1, zs), np.where(step, p1, ps), np.where(step, g1, gs)
            steps += ok
            dh = pot(zs) + 0.5 * (minv * ps * ps).sum(1) - h0
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
                    ok &= ((r * minv * ck_p[k]).sum(1) > 0) & ((r * minv * ps).sum(1) > 0)
        merge = ok
        mf, mb = (merge & fwd)[:, None], (merge & ~fwd)[:, None]
        zp, pp, gp = np.where(mf, zs, zp), np.where(mf, ps, pp), np.where(mf, gs, gp)
        zm, pm, gm = np.where(mb, zs, zm), np.where(mb, ps, pm), np.where(mb, gs, gm)
        take = merge & (np.log(RNG.random(c)) < s_lw - lw)
        prop = np.where(take[:, None], s_prop, prop)
        lw = np.where(merge, np.logaddexp(lw, s_lw), lw)
        rho = np.where(merge[:, None], rho + S, rho)
        depth = np.where(merge, j + 1, depth)
        live = merge & ((rho * minv * pm).sum(1) > 0) & ((rho * minv * pp).sum(1) > 0)
        if not live.any():
            break
    if draw == "end":
        late = (RNG.random(c) < 0.5)[:, None]
        ze, pe = np.where(late, zp, zm), np.where(late, pp, pm)
        dh = pot(ze) + 0.5 * (minv * pe * pe).sum(1) - h0
        acc = np.isfinite(dh) & (np.log(RNG.random(c)) < -np.nan_to_num(dh, nan=np.inf))
        prop = np.where(acc[:, None], ze, z)
    return prop, depth, steps, diverged


def check_on_normal(draw, mass, chains=4000, n=200, burn=50, eps=0.2):
    """Ten-dimensional normal, scales 0.5 to 5: variance over truth, mean over scale, depth.

    mass = "unit" keeps minv at one; "exact" sets it to the true variances, so
    every coordinate is a unit clock and the tree should stop at one depth.
    """
    s = np.linspace(0.5, 5.0, 10)
    pg = (lambda z: 0.5 * ((z / s) ** 2).sum(1), lambda z: z / s**2)
    minv = np.ones((chains, 10)) if mass == "unit" else np.tile(s**2, (chains, 1))
    z = s * es.RNG.standard_normal((chains, 10))
    sm1 = np.zeros(10)
    sm2 = np.zeros(10)
    dep = 0.0
    for t in range(burn + n):
        z, dp, _, _ = nuts_mass(z, eps, minv, draw=draw, pot_grad=pg)
        if t >= burn:
            sm1 += z.mean(0)
            sm2 += (z * z).mean(0)
            dep += dp.mean()
    var = sm2 / n - (sm1 / n) ** 2
    return var / s**2, sm1 / n / s, dep / n


def sbc_nuts(z, truth, minv, draw):
    """BURN transitions from the warmed state, then L with the ranks, pilots, depth, steps and divergences."""
    r = z.shape[0]
    for _ in range(BURN):
        z, _, _, _ = nuts_mass(z, EPS, minv, draw=draw)
    below = np.zeros((r, 4), dtype=np.int32)
    kept = np.empty((L, LAG_CHAINS, 3))
    depth = np.zeros(MAX_DEPTH + 1)
    steps = np.zeros(r)
    div = np.zeros(r)
    moved = 0.0
    for k in range(L):
        zn, dp, st, dv = nuts_mass(z, EPS, minv, draw=draw)
        moved += (zn != z).any(1).mean()
        z = zn
        depth += np.bincount(dp, minlength=MAX_DEPTH + 1)
        steps += st
        div += dv
        p = params("noncentred", z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, kept, depth / depth.sum(), steps / L, div / L, moved / L


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    print("0. ten-dimensional normal, scales 0.5 to 5, eps 0.2, 4000 chains x 200")
    for draw, mass in (("multinomial", "unit"), ("multinomial", "exact"), ("end", "unit"), ("end", "exact")):
        vr, mz, dep = check_on_normal(draw, mass)
        print(f"  {mass:5s} mass, {draw:11s} draw  variance / truth {vr.min():.3f} to {vr.max():.3f}  "
              f"mean / scale within {np.abs(mz).max():.3f}  mean depth {dep:.2f}  {time.time() - t0:.0f} s", flush=True)

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    truth, y = draw_studies(r)
    es.Y = y
    print(f"\nmass from warmup, day 2: non-centred nuts at eps {EPS}, max depth {MAX_DEPTH}, {STUDIES} x {nmax} "
          f"studies, L = {L}, day 1's warmup ({WARM_COST} gradients), burn {BURN} nuts transitions")

    z0, minv, _ = m1.warmup(r)
    one = np.ones_like(minv)
    print(f"\n1. the inverse mass after warmup, median over chains (10-90%)  ({time.time() - t0:.0f} s)")
    print("  " + "  ".join(
        f"{m1.NAMES[j]} {np.median(minv[:, j]):.2f} ({np.quantile(minv[:, j], 0.1):.2f}-{np.quantile(minv[:, j], 0.9):.2f})"
        for j in (0, 1, 2, 3)))

    runs = {}
    print("\n2. steps a transition (mean, median, 90th), length in units, depth shares 3-8, divergences, "
          "share of transitions that moved, lag-1 of mu / log tau / eta_1")
    for mass, draw in RUNS:
        below, kept, dshare, steps, div, moved = sbc_nuts(z0.copy(), truth, minv if mass == "adapted" else one, draw)
        runs[mass, draw] = (below.reshape(STUDIES, nmax, 4), steps, div)
        rho = m1.lag1(kept)
        print(f"  {mass:7s} {draw:11s}  steps {steps.mean():5.1f} {np.median(steps):5.1f} {np.quantile(steps, 0.9):5.1f}  "
              f"T {EPS * steps.mean():4.1f}  depth " + " ".join(f"{x:.2f}" for x in dshare[3:])
              + f"  div {100 * div.mean():.2f}%  moved {moved:.3f}  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f}  "
              f"{time.time() - t0:5.0f} s", flush=True)

    print(f"\n3. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for (mass, draw), (rr, _, _) in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  {mass:7s} {draw:11s}  {cells}")

    print(f"\n4. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for (mass, draw), (rr, _, _) in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  {mass:7s} {draw:11s}  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n5. pooled rank histograms of mu, log tau and eta_1 in tenths at n = {nmax}")
    for (mass, draw), (rr, _, _) in runs.items():
        print(f"  {mass:7s} {draw:11s}  mu {shape(rr, 0)}   log tau {shape(rr, 1)}   eta_1 {shape(rr, 3)}")

    print("\n6. where the length goes: mean steps by the study's true tau, below 1 / 1-5 / above 5")
    tau = np.exp(truth[:, 1])
    bins = [tau < 1, (tau >= 1) & (tau < 5), tau >= 5]
    for (mass, draw), (_, steps, _) in runs.items():
        print(f"  {mass:7s} {draw:11s}  steps " + " / ".join(f"{steps[b].mean():5.1f}" for b in bins))

    print(f"\n7. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}, gradients a "
          f"simulation = mean steps x ({BURN} + {L}), plus the warmup's {WARM_COST} under the adapted mass")
    for (mass, draw), (rr, steps, _) in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        cost = steps.mean() * (BURN + L) + (WARM_COST if mass == "adapted" else 0)
        print(f"  {mass:7s} {draw:11s}  band {share:.2f}  {'pass' if share <= PASS else 'fail'}  "
              f"worst {worst:8s} {steps.mean() * L:6.0f} sampling, {cost:6.0f} in all")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
