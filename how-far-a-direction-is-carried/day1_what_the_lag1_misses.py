"""Day 1 - What the lag-1 misses.

The last project's day 4 ordered its six samplers by the band's share on log
tau and the lag-1 did not: one unit at alpha 0.95 had the lowest lag-1 of
all, +0.06, and failed log tau on 0.175 of 120 studies, twice alpha 0.8's
0.075 at +0.09. Guess written before this run: the integrated
autocorrelation time, which counts every lag and not just the first, puts
the six in the band's order, and 0.95's slow carried direction shows up as
a long positive tail after the antithetic first lag.

Long chains instead of SBC: 1000 chains, each on its own study drawn from
the prior, the same per-chain mass after the 25 + 50 + 100 windows, 50 burn,
then 600 kept draws of every variant from one warmed state. IACT per
pilot from Geyer's initial positive sequence on the autocorrelation pooled
over chains (each chain centred on its own mean), and the spectrum at zero
from batch means of 50 as a second reading. The band shares are day 4's,
not rerun.

0. On the ten-dimensional normal under the exact mass every variant is
   antithetic and Geyer cannot see it: the second pair of lags is already
   negative, so the sum stops at lag 2 and reads 0.70 to 1.05 for all six.
   Batch means reads what is there, 0.89 for the tree and 0.96 for plain HMC
   against 0.13 at one unit and 0.95, 0.26 to 0.35 for the rest. Geyer's sum
   can only be trusted on a coordinate that is positively correlated.

1. On eight schools the two readings agree to within 0.3 on every
   cell. mu is antithetic under every refresh, IACT 0.14 to 0.33 against the
   tree's 1.31 and plain HMC's 1.52. Log tau is the slow one everywhere:
   full tree 3.53, one unit at alpha 0 3.27, alpha 0.8 1.42, alpha 0.95
   2.49, half a unit at 0.95 1.84, two units at 0.8 1.78. The lag-1 on log
   tau is +0.08 to +0.11 for all four refreshes, and the sum at 0.95 runs
   to lag 16 where 0.8 stops at 6, so the guess about the tail is right:
   0.95 carries a direction long enough that the small positive lags add
   up to 3.5 times 0.8's excess over one.

2. IACT puts the six in the band's order on log tau, Spearman +0.94 against
   the lag-1's +0.43; batch means the same +0.94. The one swap is half a
   unit at 0.95 (1.84, share 0.075) against two units at 0.8 (1.78, 0.125),
   0.06 apart in IACT and 0.05 in share, which is under two binomial sd at
   120 studies. mu's IACT is below log tau's on every variant, so the worse
   of the two is log tau and the joint gets the same +0.94. So the guess
   holds: the band reads the integrated autocorrelation of its slowest
   coordinate, and day 2's sweep can be read in IACT instead of 120 studies a
   cell. Not shown: whether IACT is a function of share or only in order
   with it, six points cannot say.

NumPy only. Fixed seeds. About 185 seconds, 84 of them the normal.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tree-too-long-on-log-tau"))
import day4_partial_refresh as r4  # noqa: E402  (ghmc, VARIANTS; puts the older projects on the path)
from day4_partial_refresh import c3, p1, m1, es, draw_studies, params  # noqa: E402

es.RNG = np.random.default_rng(251)

CHAINS = 1000
BURN = 50
N = 600
BATCH = 50
# Day 4's band shares over its 120 studies, (joint, log tau), in VARIANTS order.
DAY4 = {"full tree": (0.508, 0.367), "1u a0": (0.375, 0.267), "1u a.8": (0.200, 0.075),
        "1u a.95": (0.333, 0.175), "half u a.95": (0.200, 0.075), "2u a.8": (0.225, 0.125)}


def autocorr(x, max_lag):
    """Autocorrelation of each pilot up to max_lag, each chain centred on its own mean, pooled over chains.

    x is (draws, chains, pilots). Pooling the products over chains before
    dividing is what lets 1000 draws a chain give a smooth curve out to a
    few hundred lags.
    """
    x = x - x.mean(0)
    v = (x * x).sum((0, 1))
    return np.stack([(x[k:] * x[:x.shape[0] - k]).sum((0, 1)) / v for k in range(max_lag + 1)])


def iact_geyer(rho):
    """Geyer's initial positive sequence: sum pairs rho_2m + rho_2m+1 until a pair goes negative.

    Antithetic chains have a negative lag-1, so the pairs and not the single
    lags are what has to stay positive. Also returns where the sum stopped.
    """
    out, stop = [], []
    for q in range(rho.shape[1]):
        r = rho[:, q]
        pairs = r[0:-1:2] + r[1::2]
        neg = np.flatnonzero(pairs <= 0)
        m = neg[0] if neg.size else pairs.size
        out.append(-1.0 + 2.0 * pairs[:m].sum())
        stop.append(2 * m)
    return np.array(out), np.array(stop)


def batch_iact(x, b):
    """Spectrum at zero over the variance from batch means of length b, chains pooled."""
    n = (x.shape[0] // b) * b
    x = x[:n] - x[:n].mean(0)
    means = x.reshape(n // b, b, *x.shape[1:]).mean(1)
    return b * (means * means).mean((0, 1)) / (x * x).mean((0, 1))


def run(z, minv, pilots_of, seed, n=N, burn=BURN):
    """burn then n kept draws of every variant from the same state; returns the kept pilots per variant."""
    r, k = z.shape[0], len(r4.VARIANTS)
    zs = [z.copy() for _ in range(k)]
    gens = [np.random.default_rng([seed, i]) for i in range(k)]
    ps = [gens[i].standard_normal(z.shape) / np.sqrt(minv) for i in range(k)]
    kept = [[] for _ in range(k)]
    for t in range(burn + n):
        for i, (_, steps, thin, alpha) in enumerate(r4.VARIANTS):
            if steps is None:
                zs[i] = c3.nuts_cut(zs[i], r4.EPS, minv, p1.noise(seed, t, r, z.shape[1]), p1.MAX_DEPTH)[0]
            else:
                for _ in range(thin):
                    zs[i], ps[i], _ = r4.ghmc(zs[i], ps[i], minv, steps, alpha, gens[i])
            if t >= burn:
                kept[i].append(pilots_of(zs[i]))
    return [np.stack(x) for x in kept]


def report(kept, names, max_lag=300):
    """IACT, where Geyer stopped, the batch-means reading, and the lag-1 per variant and pilot."""
    out = {}
    for (lab, *_), x in zip(r4.VARIANTS, kept):
        rho = autocorr(x, max_lag)
        tau, stop = iact_geyer(rho)
        bm = batch_iact(x, BATCH)
        out[lab] = (tau, bm, rho[1])
        cells = "  ".join(f"{nm} lag-1 {rho[1, q]:+.2f} IACT {tau[q]:6.2f} (to lag {stop[q]:3d}) bm {bm[q]:6.2f}"
                          for q, nm in enumerate(names))
        print(f"  {lab:12s} {cells}", flush=True)
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def main():
    t0 = time.time()
    np.seterr(all="ignore")

    print(f"0. ten-dimensional normal, scales 0.5 to 5, eps 0.2, exact mass, {CHAINS} chains, {N} kept")
    s = np.linspace(0.5, 5.0, 10)
    saved, saved_eps = es.MODELS["noncentred"], r4.EPS
    es.MODELS["noncentred"] = (lambda x: 0.5 * ((x / s) ** 2).sum(1), lambda x: x / s ** 2)
    r4.EPS = 0.2       # twice day 4's, so the same steps are twice the units; the normal is only a check
    try:
        z = s * np.random.default_rng(252).standard_normal((CHAINS, 10))
        report(run(z, np.tile(s ** 2, (CHAINS, 1)), lambda x: x[:, [0, 9]], 2520), ["narrowest", "widest"])
    finally:
        es.MODELS["noncentred"], r4.EPS = saved, saved_eps
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n1. non-centred eight schools at eps {r4.EPS}, per-chain mass after the 25 + 50 + 100 windows, "
          f"{CHAINS} chains each on its own study")
    _, y = draw_studies(CHAINS)
    es.Y = y
    z, minv, _ = m1.warmup(CHAINS)
    res = report(run(z, minv, lambda x: params("noncentred", x)[:, [0, 1]], 2530), ["mu", "log tau"])
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

    labs = [lab for lab, *_ in r4.VARIANTS]
    share = np.array([DAY4[lab][1] for lab in labs])
    joint = np.array([DAY4[lab][0] for lab in labs])
    print("\n2. against day 4's band shares on log tau (and the joint), Spearman over the six variants")
    for k, nm in ((0, "IACT"), (1, "batch means"), (2, "lag-1")):
        v = np.array([res[lab][k][1] for lab in labs])
        w = np.array([max(res[lab][k][0], res[lab][k][1]) for lab in labs])
        print(f"  log tau {nm:11s} vs share {spearman(v, share):+.2f}   worse of mu / log tau vs joint "
              f"{spearman(w, joint):+.2f}")
    for lab in labs:
        print(f"  {lab:12s} share {DAY4[lab][1]:.3f}  IACT {res[lab][0][1]:6.2f}  lag-1 {res[lab][2][1]:+.2f}")
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
