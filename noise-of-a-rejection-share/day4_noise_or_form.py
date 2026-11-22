"""Day 4 - Noise or form.

The mass matrix project left NUTS failing on log tau after either warmup
and put it down to the per-chain mass, each chain's variance estimated
from its own few hundred draws. This file takes three fresh sets of 40
non-centred studies, 120 in all, past the 103 day 3 said a pass against a
fail needs, warms them once with a snapshot after every window, and runs
the tree from the same states under six masses: per-chain after the 100
and 400 windows, the same averaged over each study's 500 chains after 25,
100 and 400, and the prior's variances with nothing estimated. Written
before the run: if the failure is the mass's noise, the pooled masses
pass and the per-chain ones do not; the prior mass sits with the pooled.

1. Pooling takes the noise out and little else. A chain's log tau inverse
   mass after 100 runs 0.33 to 1.65 (10-90%), a study's pooled one 0.83 to
   0.98, and 1.06 to 1.16 after 400. mu's pooled value is the per-chain
   median to 0.3 (9.6, 10.9), so the form is the same and only the spread
   is gone. The three sets agree on every median to 0.04.

2. Nothing passes on 120 studies. Joint 0.400 per-chain 100, 0.408
   per-chain 400, 0.367 / 0.325 / 0.358 pooled 100 / 400 / 25, 0.333 prior,
   binomial sd 0.043-0.045, every one at least 1.9 sd above the line (P
   0.029 at best, pooled 400). Wrong: the pooled masses were to pass.

3. What pooling does buy is on log tau, and not enough. Log tau alone
   falls from 0.325 per-chain 100 to 0.183-0.233 under every pooled mass
   and the prior, a gap of 0.09 to 0.14, about 1.3 to 2 of its sd at day
   3's ratio 3.32. Its lag-1 hardly moves, +0.41 to +0.32-0.37, under any
   mass. So the noise is part of the band's failure on log tau, not the
   lag: the tree is as slow on log tau under a clean mass as a noisy one.
   Form, mostly.

4. The longer warmup buys nothing under the tree. Per-chain 400 against
   per-chain 100 is +0.008 joint, -0.033 on log tau, five steps shorter
   (35.0 against 40.3), as the mass matrix day 4 found on one set. Pooled
   25, the noisiest window with the noise averaged out, is as good as
   pooled 100 (0.358 against 0.367) and costs 50 steps a draw against 38,
   because mu's 2.7 is a quarter of its variance and its lag-1 is +0.37.

5. The prior mass is the cheapest of the six and as good as any. 0.333
   joint, log tau 0.233, 31 steps, and only the fast window before it. mu's
   25 against the posterior's 10.8 is too light by half, and its lag-1 is
   -0.30, antithetic, without hurting its band (0.05-0.10 alone). Right:
   it sits with the pooled ones.

6. Paired is not paired here. The same studies and the same warmed states
   under two masses still differ by -0.20 to +0.125 a set, because the
   chains part after one transition; the gaps are no tighter than two
   independent sets would be, so the six are read on their pooled 120.

The project's answer: at 40 studies a share is good to 0.07 one way or
the other, 103 tell 0.25 from 0.40, and on 120 the tree's log tau failure
survives every diagonal mass tried. The mass's noise was a third to a half
of log tau's excess over 0.05 and none of its lag.

NumPy only. Fixed seeds. About 6100 seconds.
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day2_share_of_a_real_sampler as s2  # noqa: E402  (read, binom_sd; puts the mass matrix project on the path)
import day3_how_many_studies as s3  # noqa: E402  (studies_needed, _tail)
from day2_share_of_a_real_sampler import m1, m2, d1, d2, es, KEYS, PARAMS, draw_studies, binom_sd  # noqa: E402
import day4_what_the_warmup_costs as w4  # noqa: E402  (the warmup with a snapshot after every window, to 400)

# Every module above reseeds on import; reseed again so these sets of studies are fresh.
es.RNG = np.random.default_rng(201)
d1.RNG = np.random.default_rng(202)
d2.RNG = np.random.default_rng(203)
m2.RNG = np.random.default_rng(204)

L, SIZES = d1.L, d1.SIZES
STUDIES, PASS, BURN = m1.STUDIES, m1.PASS, m1.BURN
SETS = 3                          # 120 studies, past the 103 day 3 said a pass against a fail needs
LOG_TAU_RATIO = 3.32              # day 3's variance ratio for NUTS's log tau alone, the share that has the extra spread
# Prior variances in the non-centred coordinates: mu ~ N(0, 5^2); log of a half-normal has variance pi^2 / 8 whatever
# its scale; every eta is N(0, 1). This mass costs nothing but the fast window.
PRIOR_MINV = np.array([es.MU_SD ** 2, math.pi ** 2 / 8] + [1.0] * es.J)
# (label, snapshot after this many slow windows, which mass). Snapshot 0 is the fast window alone.
RUNS = [
    ("per-chain 100", 3, "chain"),
    ("pooled 100", 3, "pooled"),
    ("per-chain 400", 5, "chain"),
    ("pooled 400", 5, "pooled"),
    ("pooled 25", 1, "pooled"),
    ("prior", 0, "prior"),
]


def pooled(minv):
    """Each study's inverse mass averaged over its 500 chains and handed back to every one of them.

    The chains of a study see different data, so this keeps the per-chain
    mass's average form and throws away its noise: 500 windows where a chain
    had one.
    """
    nmax = SIZES[-1]
    m = minv.reshape(STUDIES, nmax, -1).mean(1)
    return np.repeat(m, nmax, axis=0)


def mass_for(kind, minv, r):
    """The (r, 10) inverse mass a run hands the tree: the chain's own, its study's average, or the prior's."""
    if kind == "chain":
        return minv
    if kind == "pooled":
        return pooled(minv)
    return np.tile(PRIOR_MINV, (r, 1))


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    stats, crits = {}, {}
    for n in s2.SHOW:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    print(f"noise or form: NUTS (multinomial draw) on non-centred eight schools at eps {m1.EPS}, {SETS} sets of "
          f"{STUDIES} x {nmax} simulations, the same studies and the same warmed states under every mass; band on "
          f"{' / '.join(PARAMS)}; a pass is a joint share at or under {PASS}")
    print(f"prior inverse mass: mu {PRIOR_MINV[0]:.1f}, log tau {PRIOR_MINV[1]:.3f}, etas 1")

    res = {lab: {"joint": [], "alone": [], "steps": [], "div": [], "lag": []} for lab, _, _ in RUNS}
    for rep in range(SETS):
        truth, y = draw_studies(r)
        es.Y = y
        snaps = w4.warmup_snapshots(r)
        print(f"\nset {rep + 1}. inverse mass, median over chains (10-90%) for per-chain, median over studies for "
              f"pooled  ({time.time() - t0:.0f} s)")
        for k in (1, 3, 5):
            mv = snaps[k][1]
            pm = mv.reshape(STUDIES, nmax, -1).mean(1)
            print(f"  after {w4.WINDOWS[k - 1]:3d}  mu {w4.spread(mv, 0)}  log tau {w4.spread(mv, 1)}  "
                  f"pooled mu {np.median(pm[:, 0]):5.2f} log tau {np.median(pm[:, 1]):5.2f} "
                  f"(studies {pm[:, 1].min():.2f}-{pm[:, 1].max():.2f})")
        for lab, k, kind in RUNS:
            z, minv, _ = snaps[k]
            below, kept, _, steps, div, _ = m2.sbc_nuts(z.copy(), truth, mass_for(kind, minv, r), "multinomial")
            out = s2.read(below, stats, crits)[nmax]
            lag = m1.lag1(kept)
            for key, v in (("joint", out[0]), ("alone", out[1]), ("steps", steps.mean()), ("div", div.mean()),
                           ("lag", lag)):
                res[lab][key].append(v)
            print(f"  {lab:14s} joint {out[0]:.3f}  alone " + " ".join(f"{a:.3f}" for a in out[1])
                  + f"  steps {steps.mean():5.1f}  div {100 * div.mean():.2f}%  lag-1 mu {lag[0]:+.2f} "
                  f"log tau {lag[1]:+.2f}  {time.time() - t0:5.0f} s", flush=True)

    n_all = SETS * STUDIES
    base = np.array(res["per-chain 100"]["joint"])
    base_lt = np.array([a[1] for a in res["per-chain 100"]["alone"]])
    print(f"\npooled over the {n_all} studies: joint share (binomial sd), log tau alone (sd at day 3's ratio "
          f"{LOG_TAU_RATIO}), the chance a sampler on {PASS} reads this low, and the paired gap to per-chain 100")
    for lab, _, _ in RUNS:
        g = np.array(res[lab]["joint"])
        lt = np.array([a[1] for a in res[lab]["alone"]])
        p, plt = g.mean(), lt.mean()
        sd = binom_sd(p, n_all)
        sdl = binom_sd(plt, n_all) * math.sqrt(LOG_TAU_RATIO)
        tail = s3._tail(abs(p - PASS) / binom_sd(PASS, n_all))
        gap, gap_lt = g - base, lt - base_lt
        verdict = "pass" if p <= PASS and tail < 0.05 else ("fail" if p > PASS and tail < 0.05 else "on the line")
        print(f"  {lab:14s} joint {p:.3f} ({sd:.3f})  log tau {plt:.3f} ({sdl:.3f})  "
              f"{'P low' if p < PASS else 'P high'} {tail:.3f}  {verdict:11s} gap {gap.mean():+.3f} "
              f"(sets {' '.join(f'{x:+.3f}' for x in gap)})  log tau gap {gap_lt.mean():+.3f}  "
              f"steps {np.mean(res[lab]['steps']):5.1f}  lag-1 log tau {np.mean([x[1] for x in res[lab]['lag']]):+.2f}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
