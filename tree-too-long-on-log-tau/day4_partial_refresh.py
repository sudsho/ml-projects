"""Day 4 - Partial refresh.

Day 3 killed the project's guess: every cut of the tree put log tau's lag
and share up, so if the length is the trouble it is too short. The other way
to a longer move is not a longer tree but a momentum that survives the
transition. Generalised HMC keeps the momentum with weight alpha and tops it
up with fresh noise, so a chain keeps going the way it was going. Fixed
lengths of half a unit to two units at alpha 0 to 0.95, each thinned so it
spends 40 gradients a kept draw, the full tree's 40.3, against the full tree
on three fresh sets of 40 studies x 500 from the same warmed state,
per-chain mass after the 100 window, 50 burn then 99. No prediction was
written before this run.

0. Every refresh keeps its target. On the ten-dimensional normal under the
   exact mass, variance over truth 0.995 to 1.008, mean within 0.003 of a
   scale. With alpha above zero the draws go antithetic on every
   coordinate, lag-1 -0.44 at alpha 0.8 and -0.78 at half a unit and 0.95.

1. Partial refresh fixes log tau at the full tree's price. Over the 120
   studies the full tree fails 0.508 joint, 0.100 mu, 0.367 log tau, lag-1
   on log tau +0.40 (day 3's 0.467 / 0.375 / +0.41 on other studies). One
   unit at alpha 0.8 fails 0.200 joint, 0.050 mu, 0.075 log tau, lag-1 +0.09,
   and half a unit at alpha 0.95 reads the same 0.200 / 0.075 at +0.07. Two
   units at 0.8 is 0.225 / 0.125. Every refresh beats the tree on every
   set, and the 0.29 off log tau is about six binomial sd at 120.

2. It is the refresh, not the fixed length. The same one unit at alpha 0,
   plain HMC thinned four, is 0.375 joint and 0.267 log tau at lag-1 +0.42,
   the tree's lag at 0.1 less share. So log tau's lag-1 of +0.4 under
   every diagonal mass in two projects is what any sampler gets that forgets
   its direction every 40 gradients, and keeping it takes the lag to +0.1.

3. More memory is not better. One unit at 0.95 has the lowest lag-1 on log
   tau (+0.06) and fails it on 0.175, twice 0.8's, with mu antithetic at
   -0.57. A lag-1 near zero is not the whole of the autocorrelation; at 0.95
   the chain carries one direction for about ten transitions, and the band
   sees the slow part the lag-1 does not.

4. The price. 8460 gradients a study for the refresh, 2500 of them the
   warmup, against 8500 for the full tree. The best joint, 0.200, sits under
   unit mass at four units (0.223 over 1280 studies, 6710 gradients) and the
   two units after the 400 window (0.23, 11480), and is 1.5 binomial sd above
   a correct sampler's 0.143 to 0.185, so on 120 studies it passes the line
   and is not shown to be nominal.

NumPy only. Fixed seeds. About 1560 seconds, 14 of them the normal.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_paired_readings as p1  # noqa: E402  (noise, nuts_crn; puts the older projects on the path)
import day3_the_tree_cut_short as c3  # noqa: E402  (nuts_cut, the full tree)
from day1_paired_readings import m1, d1, d2, es, KEYS, draw_studies, params  # noqa: E402

# day 3 reseeds on import; reseed again so these 120 studies are fresh.
es.RNG = np.random.default_rng(241)
d1.RNG = np.random.default_rng(242)
d2.RNG = np.random.default_rng(243)

L, SIZES, EPS, STUDIES, BURN, LAG_CHAINS = p1.L, p1.SIZES, p1.EPS, p1.STUDIES, p1.BURN, p1.LAG_CHAINS
SETS = 3
WARM_GRADS = (m1.FAST + sum(m1.WINDOWS)) * m1.WARM_STEPS   # 2500, the 25 + 50 + 100 warmup every variant shares
# Each variant is (label, leapfrog steps a transition, transitions a kept draw, alpha), or None for the full tree.
# Every fixed variant spends 40 gradients a kept draw, the full tree's 40.3 to within a step.
VARIANTS = [("full tree", None, None, None),
            ("1u a0", 10, 4, 0.0), ("1u a.8", 10, 4, 0.8), ("1u a.95", 10, 4, 0.95),
            ("half u a.95", 5, 8, 0.95), ("2u a.8", 20, 2, 0.8)]


def ghmc(z, p, minv, steps, alpha, g):
    """One generalised HMC transition (Horowitz) under a per-chain diagonal mass.

    The momentum is kept with weight alpha and topped up with fresh N(0, M)
    noise, p <- alpha p + sqrt(1 - alpha^2) xi, which leaves N(0, M) alone.
    Then `steps` leapfrogs and a Metropolis test on the joint (z, p); an
    accepted move keeps its end momentum, a rejected one stays put and flips
    the momentum. Each of the three pieces leaves the joint target alone, so
    the chain does, and with alpha near one it keeps going the way it was
    going instead of picking a new direction every transition.
    """
    pot, grad = es.MODELS["noncentred"]
    p = alpha * p + np.sqrt(1.0 - alpha * alpha) * g.standard_normal(z.shape) / np.sqrt(minv)
    h0 = pot(z) + 0.5 * (minv * p * p).sum(1)
    zn, pn = z.copy(), p.copy()
    gr = grad(zn)
    for _ in range(steps):
        pn = pn - 0.5 * EPS * gr
        zn = zn + EPS * minv * pn
        gr = grad(zn)
        pn = pn - 0.5 * EPS * gr
    dh = pot(zn) + 0.5 * (minv * pn * pn).sum(1) - h0
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(g.random(z.shape[0])) < -np.nan_to_num(dh, nan=np.inf))
    a = acc[:, None]
    return np.where(a, zn, z), np.where(a, pn, -p), acc


def check_on_normal(chains=4000, n=200, burn=50, eps=0.2):
    """The ten-dimensional normal of day 1 under the exact mass: does every refresh keep its target."""
    global EPS
    s = np.linspace(0.5, 5.0, 10)
    saved, saved_eps = es.MODELS["noncentred"], EPS
    es.MODELS["noncentred"] = (lambda x: 0.5 * ((x / s) ** 2).sum(1), lambda x: x / s ** 2)
    EPS = eps
    exact = np.tile(s ** 2, (chains, 1))
    out = {}
    try:
        for lab, steps, thin, alpha in VARIANTS[1:]:
            g = np.random.default_rng([2440, steps, thin, int(alpha * 100)])
            z = s * np.random.default_rng(244).standard_normal((chains, 10))
            p = g.standard_normal(z.shape) / np.sqrt(exact)
            draws = []
            for t in range(burn + n):
                for _ in range(thin):
                    z, p, _ = ghmc(z, p, exact, steps // 2, alpha, g)   # eps 0.2, so half the steps for the same units
                if t >= burn:
                    draws.append(z)
            x = np.stack(draws)
            out[lab] = (x.var((0, 1)) / s ** 2, x.mean((0, 1)) / s, m1.lag1(x[:, :, [0, 9]]))
    finally:
        es.MODELS["noncentred"], EPS = saved, saved_eps
    return out


def sbc_variants(z, truth, minv, seed):
    """BURN then L kept draws of every variant from the same warmed state; ranks, pilots, cost, acceptance."""
    r, k = z.shape[0], len(VARIANTS)
    zs = [z.copy() for _ in range(k)]
    gens = [np.random.default_rng([seed, i]) for i in range(k)]
    ps = [gens[i].standard_normal(z.shape) / np.sqrt(minv) for i in range(k)]
    below = [np.zeros((r, 4), np.int32) for _ in range(k)]
    kept = [np.empty((L, LAG_CHAINS, 3)) for _ in range(k)]
    grads = np.zeros(k)
    acc = np.zeros(k)
    for t in range(BURN + L):
        for i, (_, steps, thin, alpha) in enumerate(VARIANTS):
            if steps is None:
                zs[i], s_, _, _ = c3.nuts_cut(zs[i], EPS, minv, p1.noise(seed, t, r, z.shape[1]), p1.MAX_DEPTH)
                g_, a_ = s_.mean(), 1.0
            else:
                a_ = 0.0
                for _ in range(thin):
                    zs[i], ps[i], ok = ghmc(zs[i], ps[i], minv, steps, alpha, gens[i])
                    a_ += ok.mean() / thin
                g_ = steps * thin
            if t >= BURN:
                grads[i] += g_ / L
                acc[i] += a_ / L
                p = params("noncentred", zs[i])
                kept[i][t - BURN] = p[:LAG_CHAINS][:, [0, 1, 3]]
                below[i] += p < truth
    return below, kept, grads, acc


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    print("0. ten-dimensional normal, scales 0.5 to 5, eps 0.2, exact mass, 4000 chains, 200 kept")
    for lab, (vr, mz, lg) in check_on_normal().items():
        print(f"  {lab:12s} variance / truth {vr.min():.3f} to {vr.max():.3f}, mean / scale within {np.abs(mz).max():.3f}, "
              f"lag-1 narrowest {lg[0]:+.2f} widest {lg[1]:+.2f}")
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

    s, _ = d2.shape_stats(nmax)
    crits = {k: v for k, v in d2.critical_values(dict(s, n=nmax)).items() if k in KEYS}
    band, crit = s["band"], crits["band"]

    print(f"\npartial refresh: non-centred at eps {EPS}, per-chain mass from the 25 + 50 + 100 windows, "
          f"{SETS} sets of {STUDIES} x {nmax}, every variant from the same warmed state")
    rec = {lab: {"joint": [], "mu": [], "log tau": [], "lag": [], "grads": [], "acc": []} for lab, *_ in VARIANTS}
    for rep in range(SETS):
        truth, y = draw_studies(r)
        es.Y = y
        z, minv, _ = m1.warmup(r)
        below, kept, grads, acc = sbc_variants(z, truth, minv, 2450 + rep)
        print(f"\nset {rep + 1}  ({time.time() - t0:.0f} s)")
        for i, (lab, *_) in enumerate(VARIANTS):
            rr = below[i].reshape(STUDIES, nmax, 4)
            alone = np.stack([band(d1.rank_counts(rr[:, :, q])) > crit for q in range(4)], 1)
            lag = m1.lag1(kept[i])
            for key, v in (("joint", alone.any(1)), ("mu", alone[:, 0]), ("log tau", alone[:, 1]), ("lag", lag),
                           ("grads", grads[i]), ("acc", acc[i])):
                rec[lab][key].append(v)
            print(f"  {lab:12s} joint {alone.any(1).mean():.3f}  mu {alone[:, 0].mean():.3f}  log tau {alone[:, 1].mean():.3f}  "
                  f"grads {grads[i]:5.1f}  acc {acc[i]:.3f}  lag-1 mu {lag[0]:+.2f} log tau {lag[1]:+.2f}", flush=True)

    n = SETS * STUDIES
    print(f"\nover the {n} studies; gradients a study = warmup {WARM_GRADS} + {BURN + L} draws")
    full = {k: np.concatenate(rec["full tree"][k]) for k in ("joint", "mu", "log tau")}
    for lab, *_ in VARIANTS:
        x = {k: np.concatenate(rec[lab][k]) for k in ("joint", "mu", "log tau")}
        lag = np.mean(rec[lab]["lag"], 0)
        cost = WARM_GRADS + (BURN + L) * np.mean(rec[lab]["grads"])
        line = f"  {lab:12s} {cost:6.0f} grads  lag-1 mu {lag[0]:+.2f} log tau {lag[1]:+.2f}"
        for k in ("joint", "mu", "log tau"):
            gap = x[k].mean() - full[k].mean()
            line += f" | {k} {x[k].mean():.3f} ({gap:+.3f})"
        print(line)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
