"""Day 2 - The share of a real sampler.

Day 1 found the band's share binomial when the ranks are uniform by
construction, and left one test: if a real sampler's share over fresh sets
of 40 studies spreads wider than the binomial at its own mean, something is
shared between the studies of a set. This file runs the two samplers the
last project leaned on, each on fresh sets of 40 non-centred eight schools
studies of 500 simulations: unit mass at four units after the fast window,
which passed on both sets of the mass matrix day 4 (0.25, 0.17), on 24
sets, and NUTS under the 100 window mass, which failed there (0.47, 0.42),
on 6, since a NUTS set costs eight times as much. No prediction was written
for the numbers; the day 1 test is the only claim going in.

1. The passing sampler sits on the line. Unit mass at four units reads a
   joint 0.241 at 500, against 0.185 for four tests at 5%, with mu and
   log tau each at 0.08 alone. 9 of 24 sets are above 0.25 and the range
   is 0.125 to 0.375. The last project's two readings were two of these.
   So "passes" was a coin with a 38% chance of saying fail, not a pass.

2. Its spread is the binomial's, a little under. sd 0.058 over sets against
   0.068 at its mean, variance ratio 0.74, and a binomial set of 24 sets is
   as spread 82% of the time. Each parameter alone 0.62 to 1.34, none with
   a tail under 0.14. Nothing shared, the way day 1 found for the null.

3. The failing sampler fails on every set, and its log tau is not binomial.
   NUTS reads 0.275 to 0.500, mean 0.396, none at or under 0.25, joint sd
   0.083 against 0.077 (ratio 1.15, P 0.34). But log tau alone is 0.075 to
   0.400 with sd 0.113 against 0.072, ratio 2.48, and a binomial gives a
   spread that wide on 3.0% of six-set draws. One set in six read log tau
   at 0.075, a clean reading of a sampler that fails on it. With six sets
   this is one tail, not a measurement; the joint hides it because the
   other three parameters are narrower than binomial (ratios 0.27 to 1.58).

4. Two readings of one sampler. Pairs of unit-mass sets differ by a median
   0.050 and 0.18 or more on 2% of pairs; NUTS pairs by 0.100 and 0.18 or
   more on 13%, max 0.225. The 0.60 against 0.42 from the mass matrix
   project is a NUTS-sized gap, and 0.60 is above every one of these six.

5. At 100 simulations the two cannot be told apart. Unit mass 0.217 (sd
   0.069), NUTS 0.233 (sd 0.026), three unit sets above NUTS's worst. Every
   verdict needed the 500.

What this leaves for day 3: a binomial at 40 studies cannot tell 0.24 from
0.40 on one set with any comfort (the gap is 2.0 of the larger sd), and
log tau's extra spread under NUTS, if it is real, means more studies help
less there than the binomial says. The six NUTS sets are the weak count.

NumPy only. Fixed seeds. About 2830 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mass-matrix-from-warmup"))
import day1_diagonal_mass_sweep as m1  # noqa: E402  (fixed-length HMC under a mass, the windows; puts the older projects on the path)
import day2_nuts_under_the_mass as m2  # noqa: E402  (the tree under a diagonal mass)
from day1_diagonal_mass_sweep import d1, d2, es, per_param, KEYS, PARAMS, draw_studies  # noqa: E402

# Every module above reseeds on import; reseed again so these sets of studies are fresh.
es.RNG = np.random.default_rng(181)
d1.RNG = np.random.default_rng(182)
d2.RNG = np.random.default_rng(183)
m2.RNG = np.random.default_rng(184)

L, SIZES = d1.L, d1.SIZES
STUDIES, PASS, BURN = m1.STUDIES, m1.PASS, m1.BURN
FAST, WARM_STEPS = m1.FAST, m1.WARM_STEPS
UNIT_STEPS = 40                  # four units at eps 0.1, the unit-mass pass of the last two projects
SETS_UNIT = 24                   # fresh sets of 40 studies for the passing sampler, about 40 s a set
SETS_NUTS = 6                    # and for the failing one, about 340 s a set
SHOW = (100, 500)                # sizes printed per set; every verdict was read at 500


def binom_sd(p, n=STUDIES):
    """sd of a share of n independent studies each rejected with probability p."""
    return np.sqrt(p * (1 - p) / n)


def unit_set(truth):
    """Unit mass at four units after the fast window alone, the way the mass matrix day 4 ran it."""
    r = truth.shape[0]
    z = es.prior_start("noncentred", r)
    minv = np.ones((r, 10))
    for _ in range(FAST):
        z, _ = m1.hmc_mass(z, WARM_STEPS, minv)
    below, _, _ = m1.sbc_mass(z, truth, minv, UNIT_STEPS)
    return below


def nuts_set(truth):
    """NUTS with the multinomial draw under the mass from the 25 + 50 + 100 windows."""
    z, minv, _ = m1.warmup(truth.shape[0])
    below, _, _, steps, _, _ = m2.sbc_nuts(z, truth, minv, "multinomial")
    return below, steps.mean()


def read(below, stats, crits):
    """Joint band share and the share on each parameter alone, at every size in SHOW."""
    nmax = SIZES[-1]
    rr = below.reshape(STUDIES, nmax, 4)
    out = {}
    for n in SHOW:
        sub = rr[:, :n]
        out[n] = (d2.reject_any(sub, stats[n], crits[n])["band"], per_param(sub, stats[n], crits[n], "band"))
    return out


def dispersion(x, p):
    """Between-set variance over the binomial's at the pooled share, and the chance a binomial gives one as large.

    (k - 1) s^2 / (p (1 - p) / 40) is chi-square with k - 1 degrees of freedom
    when the 40 studies of every set are independent draws at one chance p; the
    tail comes from 20,000 binomial sets of the same size, so it needs no SciPy.
    """
    k = len(x)
    ratio = x.var(ddof=1) / binom_sd(p) ** 2 if 0 < p < 1 else np.nan
    sims = np.random.default_rng(185).binomial(STUDIES, p, (20000, k)) / STUDIES
    tail = (sims.var(1, ddof=1) >= x.var(ddof=1) - 1e-12).mean()
    return ratio, tail


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    stats, crits = {}, {}
    for n in SHOW:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    print(f"the share of a real sampler: non-centred eight schools at eps {m1.EPS}, sets of {STUDIES} x {nmax} studies, "
          f"L = {L}, burn {BURN}, band on {' / '.join(PARAMS)}; a pass is a joint share at or under {PASS}")

    runs = {"unit 4.0": [], "nuts 100": []}
    print(f"\n1. unit mass at {UNIT_STEPS * m1.EPS:.1f} units after the fast window, {SETS_UNIT} sets")
    for rep in range(SETS_UNIT):
        truth, y = draw_studies(r)
        es.Y = y
        res = read(unit_set(truth), stats, crits)
        runs["unit 4.0"].append(res)
        print(f"  set {rep + 1:2d}  " + "  ".join(
            f"n={n} joint {res[n][0]:.3f} alone " + " ".join(f"{a:.3f}" for a in res[n][1]) for n in SHOW)
            + f"  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n2. NUTS under the 100 window mass, {SETS_NUTS} sets")
    for rep in range(SETS_NUTS):
        truth, y = draw_studies(r)
        es.Y = y
        below, steps = nuts_set(truth)
        res = read(below, stats, crits)
        runs["nuts 100"].append(res)
        print(f"  set {rep + 1:2d}  " + "  ".join(
            f"n={n} joint {res[n][0]:.3f} alone " + " ".join(f"{a:.3f}" for a in res[n][1]) for n in SHOW)
            + f"  steps {steps:4.1f}  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n3. between sets at n = {nmax}: mean, sd over sets against the binomial at that mean, the variance ratio "
          f"and the chance a binomial set of sets is as spread, range, sets on the far side of {PASS}")
    for name, rows in runs.items():
        g = np.array([res[nmax][0] for res in rows])
        a = np.array([res[nmax][1] for res in rows])
        ratio, tail = dispersion(g, g.mean())
        far = (g > PASS).mean() if name.startswith("unit") else (g <= PASS).mean()
        print(f"  {name}  joint {g.mean():.3f}  sd {g.std(ddof=1):.3f} ({binom_sd(g.mean()):.3f})  ratio {ratio:4.2f}  "
              f"P {tail:.3f}  range {g.min():.3f}-{g.max():.3f}  far side {far:.2f}")
        for p, pname in enumerate(PARAMS):
            ratio, tail = dispersion(a[:, p], a[:, p].mean())
            print(f"    {pname:8s} alone {a[:, p].mean():.3f}  sd {a[:, p].std(ddof=1):.3f} "
                  f"({binom_sd(a[:, p].mean()):.3f})  ratio {ratio:4.2f}  P {tail:.3f}")
        # every pair of sets is two readings of one sampler, the way the last project read 0.60 and 0.42
        d = np.abs(g[:, None] - g[None, :])[np.triu_indices(len(g), 1)]
        print(f"    pairs of sets: median gap {np.median(d):.3f}, 90% {np.quantile(d, 0.9):.3f}, "
              f"max {d.max():.3f}, share of pairs 0.18 or more apart {(d >= 0.18 - 1e-9).mean():.2f}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
