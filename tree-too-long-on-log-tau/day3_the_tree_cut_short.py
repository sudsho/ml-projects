"""Day 3 - The tree cut short.

Day 2 found the draw lands past log tau's turn on 0.61 of transitions and
that this does not sort rejected studies from passed ones. If overshooting
is harmless, a shorter tree should leave log tau's share where it is and
only cost mu; if it is the failure, a shorter tree should fix log tau.
So day 1's tree with a cap at 3 to 6 doublings, and the full tree with its
draw kept off the last doubling, the half the start sits in. Every variant
on the same three fresh sets of 40 studies x 500, the same warmed state and
the same numbers, per-chain mass after the 100 window, 50 burn then 99.
Written before the run: depth 5 and the first half leave log tau within
0.1 of the full tree, and mu pays for it.

0. Every variant keeps its target. On the ten-dimensional normal under the
   exact mass, variance over truth 0.993 to 1.002 and mean within 0.005 of a
   scale for all six, and the full variant is day 1's tree to the bit.

1. Wrong, and the other way round. Cutting the tree costs log tau first
   and most. Over the 120 studies the full tree fails 0.467 joint, 0.100 on
   mu, 0.375 on log tau (40.3 steps, lag-1 +0.08 / +0.41). Depth 6 barely
   binds (0.004 capped) and is identical. Depth 5 caps 0.29 of transitions,
   saves a quarter of the gradients (30.8) and puts log tau up +0.208 (sd of
   the paired gap 0.042, five sd) against mu +0.042; lag-1 on log tau +0.46.
   Depth 4 (15 steps) fails every study on log tau and 0.62 on mu, depth 3
   all of both.

2. The first half is the cleanest test, the same tree and the same 40.3
   gradients with only the draw moved toward the start, and it fails log
   tau on 0.983 against 0.375, lag-1 +0.72, while mu goes to 0.442. Day 2's
   draws past the turn are not waste. They are the part of the move log tau
   needs.

3. So the guess this project started from is dead: the tree is not too
   long on log tau. Every shortening moves log tau's lag-1 up and its share
   up, in step, and nothing shorter gets near the full tree's 0.375. If the
   lag is the tree's length at all it is too short, the trajectory
   project's mu story on another coordinate. Three sets read 0.450 / 0.250 /
   0.425 on the full tree, so a set of 40 still moves by 0.2, and the paired
   gap at 120 is what carries the verdict.

Day 4 turns to the other side: partial momentum refresh, which keeps the
direction between transitions, against the full tree, priced in gradients
against unit mass at four units and the fixed two units after the 400 window.

NumPy only. Fixed seeds. About 3000 seconds, 250 of them the normal.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_paired_readings as p1  # noqa: E402  (noise, pooled, hits; puts the older projects on the path)
from day1_paired_readings import m1, d1, d2, es, KEYS, draw_studies, params  # noqa: E402

es.RNG = np.random.default_rng(231)
d1.RNG = np.random.default_rng(232)
d2.RNG = np.random.default_rng(233)

L, SIZES, EPS, STUDIES, BURN, LAG_CHAINS = p1.L, p1.SIZES, p1.EPS, p1.STUDIES, p1.BURN, p1.LAG_CHAINS
SETS = 3                             # three fresh sets of 40, 120 studies, all variants on each
# Each variant is (label, deepest doubling allowed, draw from the first half). All share one seed per set.
VARIANTS = [("full", p1.MAX_DEPTH, False), ("depth 6", 6, False), ("depth 5", 5, False),
            ("depth 4", 4, False), ("depth 3", 3, False), ("first half", p1.MAX_DEPTH, True)]


def nuts_cut(z, eps, minv, u, depth, half=False):
    """Day 1's nuts_crn with the tree capped at `depth` doublings, or its draw kept off the last doubling.

    With half set the tree is grown exactly as the full one, U-turns and all,
    but the multinomial choice made at the last doubling that merged is
    undone, so the draw comes from the subtree the start sits in, half the
    final tree. Any point of that half grows the same full tree on the same
    coins, so the draw is still a multinomial over a set that every member of
    it would pick, and the chain keeps its target. The cap is the usual
    max-depth: a chain that is still live at `depth` stops there.

    Returns (z, steps, capped, stayed): leapfrog steps, whether the cap cut a
    live tree, and whether the draw is the start.
    """
    pot, grad = es.MODELS["noncentred"]
    c, d = z.shape
    p0 = u["xi"] / np.sqrt(minv)
    g0 = grad(z)
    h0 = pot(z) + 0.5 * (minv * p0 * p0).sum(1)
    zm, pm, gm = z.copy(), p0.copy(), g0.copy()
    zp, pp, gp = z.copy(), p0.copy(), g0.copy()
    prop, lw, rho = z.copy(), np.zeros(c), p0.copy()
    held = z.copy()                  # the draw as it stood before the last merge
    live = np.ones(c, bool)
    steps = np.zeros(c, np.int32)
    for j in range(depth):
        fwd = u["dir"][:, j] < 0.5
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
            q = ps - 0.5 * e * gs
            z1 = zs + e * minv * q
            g1 = grad(z1)
            q = q - 0.5 * e * g1
            step = ok[:, None]
            zs, ps, gs = np.where(step, z1, zs), np.where(step, q, ps), np.where(step, g1, gs)
            steps += ok
            dh = pot(zs) + 0.5 * (minv * ps * ps).sum(1) - h0
            bad = ok & (~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000))
            ok &= ~bad
            leaf = np.where(np.isfinite(dh), -dh, -np.inf)
            grown = np.logaddexp(s_lw, leaf)
            take = ok & (np.log(u["leaf"][:, 2 ** j - 1 + n]) < leaf - grown)
            s_prop = np.where(take[:, None], zs, s_prop)
            s_lw = np.where(ok, grown, s_lw)
            for k in range(1, j + 1):
                if n % 2 ** k == 0:
                    ck_p[k], ck_S[k] = ps, S
            S = S + ps
            for k in range(1, j + 1):
                if (n + 1) % 2 ** k == 0:
                    r = S - ck_S[k]
                    ok &= ((r * minv * ck_p[k]).sum(1) > 0) & ((r * minv * ps).sum(1) > 0)
        merge = ok
        held = np.where(merge[:, None], prop, held)
        mf, mb = (merge & fwd)[:, None], (merge & ~fwd)[:, None]
        zp, pp, gp = np.where(mf, zs, zp), np.where(mf, ps, pp), np.where(mf, gs, gp)
        zm, pm, gm = np.where(mb, zs, zm), np.where(mb, ps, pm), np.where(mb, gs, gm)
        take = merge & (np.log(u["merge"][:, j]) < s_lw - lw)
        prop = np.where(take[:, None], s_prop, prop)
        lw = np.where(merge, np.logaddexp(lw, s_lw), lw)
        rho = np.where(merge[:, None], rho + S, rho)
        live = merge & ((rho * minv * pm).sum(1) > 0) & ((rho * minv * pp).sum(1) > 0)
        if not live.any():
            break
    out = held if half else prop
    return out, steps, live, (out == z).all(1)


def check(z, minv):
    """The full variant is day 1's tree: same draws and steps on the same numbers."""
    u = p1.noise(2390, 0, z.shape[0], z.shape[1])
    a, _, sa, _ = p1.nuts_crn(z, EPS, minv, u)
    b, sb, _, _ = nuts_cut(z, EPS, minv, u, p1.MAX_DEPTH)
    return np.abs(a - b).max(), np.abs(sa - sb).max()


def check_on_normal(chains=4000, n=200, burn=50):
    """The ten-dimensional normal of day 1 under the exact mass: does every variant keep its target."""
    s = np.linspace(0.5, 5.0, 10)
    saved = es.MODELS["noncentred"]
    es.MODELS["noncentred"] = (lambda x: 0.5 * ((x / s) ** 2).sum(1), lambda x: x / s ** 2)
    exact = np.tile(s ** 2, (chains, 1))
    z0 = s * np.random.default_rng(234).standard_normal((chains, 10))
    out = {}
    try:
        for lab, depth, half in VARIANTS:
            z, draws = z0.copy(), []
            for t in range(burn + n):
                z = nuts_cut(z, 0.2, exact, p1.noise(2341, t, chains, 10), depth, half)[0]
                if t >= burn:
                    draws.append(z)
            x = np.stack(draws)
            out[lab] = (x.var((0, 1)) / s ** 2, x.mean((0, 1)) / s)
    finally:
        es.MODELS["noncentred"] = saved
    return out


def sbc_variants(z, truth, minv, seed):
    """BURN then L transitions of every variant from the same warmed state on the same numbers."""
    r, k = z.shape[0], len(VARIANTS)
    zs = [z.copy() for _ in range(k)]
    below = [np.zeros((r, 4), np.int32) for _ in range(k)]
    kept = [np.empty((L, LAG_CHAINS, 3)) for _ in range(k)]
    st = np.zeros(k)
    cap = np.zeros(k)
    stay = np.zeros(k)
    for t in range(BURN + L):
        u = p1.noise(seed, t, r, z.shape[1])
        for i, (_, depth, half) in enumerate(VARIANTS):
            zs[i], s_, c_, y_ = nuts_cut(zs[i], EPS, minv, u, depth, half)
            if t >= BURN:
                st[i] += s_.mean() / L
                cap[i] += c_.mean() / L
                stay[i] += y_.mean() / L
                p = params("noncentred", zs[i])
                kept[i][t - BURN] = p[:LAG_CHAINS][:, [0, 1, 3]]
                below[i] += p < truth
    return below, kept, st, cap, stay


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    print("0. ten-dimensional normal, scales 0.5 to 5, eps 0.2, exact mass, 4000 chains, 200 kept")
    for lab, (vr, mz) in check_on_normal().items():
        print(f"  {lab:10s} variance / truth {vr.min():.3f} to {vr.max():.3f}, mean / scale within {np.abs(mz).max():.3f}")
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

    s, _ = d2.shape_stats(nmax)
    stats = {k: s[k] for k in KEYS}
    crits = {k: v for k, v in d2.critical_values(dict(s, n=nmax)).items() if k in KEYS}
    band, crit = stats["band"], crits["band"]

    print(f"\nthe tree cut short: non-centred NUTS at eps {EPS}, per-chain mass from the 25 + 50 + 100 windows, "
          f"{SETS} sets of {STUDIES} x {nmax}, every variant on one seed a set")
    rec = {lab: {"joint": [], "mu": [], "log tau": [], "lag": [], "steps": [], "cap": [], "stay": []} for lab, _, _ in VARIANTS}
    for rep in range(SETS):
        truth, y = draw_studies(r)
        es.Y = y
        z, minv, _ = m1.warmup(r)
        if rep == 0:
            gz, gs = check(z, minv)
            print(f"check: full variant against day 1's tree, largest gap in the draw {gz:.1e}, in steps {gs}")
        below, kept, st, cap, stay = sbc_variants(z, truth, minv, 2350 + rep)
        print(f"\nset {rep + 1}  ({time.time() - t0:.0f} s)")
        for i, (lab, _, _) in enumerate(VARIANTS):
            rr = below[i].reshape(STUDIES, nmax, 4)
            alone = np.stack([band(d1.rank_counts(rr[:, :, p])) > crit for p in range(4)], 1)
            lag = m1.lag1(kept[i])
            for key, v in (("joint", alone.any(1)), ("mu", alone[:, 0]), ("log tau", alone[:, 1]), ("lag", lag),
                           ("steps", st[i]), ("cap", cap[i]), ("stay", stay[i])):
                rec[lab][key].append(v)
            print(f"  {lab:10s} joint {alone.any(1).mean():.3f}  mu {alone[:, 0].mean():.3f}  log tau {alone[:, 1].mean():.3f}  "
                  f"steps {st[i]:5.1f}  capped {cap[i]:.3f}  stayed {stay[i]:.3f}  "
                  f"lag-1 mu {lag[0]:+.2f} log tau {lag[1]:+.2f}", flush=True)

    n = SETS * STUDIES
    print(f"\nover the {n} studies, each variant against the full tree on the same studies and numbers")
    full = {k: np.concatenate(rec["full"][k]) for k in ("joint", "mu", "log tau")}
    for lab, _, _ in VARIANTS:
        x = {k: np.concatenate(rec[lab][k]) for k in ("joint", "mu", "log tau")}
        lag = np.mean(rec[lab]["lag"], 0)
        line = f"  {lab:10s} steps {np.mean(rec[lab]['steps']):5.1f}  lag-1 mu {lag[0]:+.2f} log tau {lag[1]:+.2f}"
        for k in ("joint", "mu", "log tau"):
            gap = x[k].mean() - full[k].mean()
            disc = (x[k] != full[k]).mean()
            sd = np.sqrt(max(disc - gap ** 2, 0.0) / n)
            line += f" | {k} {x[k].mean():.3f} ({gap:+.3f} sd {sd:.3f})"
        print(line)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
