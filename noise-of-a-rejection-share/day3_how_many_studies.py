"""Day 3 - How many studies.

Day 2 left the binomial standing for unit mass and in doubt for NUTS's
log tau, on 24 and 6 sets. This file runs fresh sets of 40 non-centred
eight schools studies of 500 simulations, 32 for unit mass at four units
and 4 for NUTS under the 100 window mass, pools them two and four at a
time into sets of 80 and 160, and reads the spread at each size, the
studies needed to tell two shares apart and what they cost, and the old
one-set verdicts against the line. The prediction going in was the
binomial's: sd falling as one over the root of the studies.

1. The binomial holds for unit mass at every size. Joint 0.223 over 32
   sets, sd 0.065 / 0.046 / 0.029 at 40 / 80 / 160 against 0.066 / 0.047
   / 0.033, so sd x sqrt(studies) stays 0.41 (0.36 at 160, on 8 sets).
   Each parameter alone 0.77 to 1.01. 8 of the 32 sets of 40 read above
   0.25, and 2 of the 8 sets of 160 still do. Day 2's 0.241 was this
   sampler too.

2. Unit mass at four units is not a null sampler either. Pooled over 1280
   studies its 0.223 sits 0.038 above the 0.185 four tests at 5% give, 3.3
   of the pooled sd. It passed the line and fails nominal, and no single set
   of 40 could have shown that.

3. NUTS's log tau is wider than binomial again. Joint 0.481 on 4 sets,
   0.400 to 0.575, ratio 1.09; log tau alone 0.225 to 0.500, ratio 3.32,
   after day 2's 2.48 on six other sets. Ten sets in all and both ratios are
   high, so the extra spread on log tau is now two readings of one thing,
   still on few sets. The other three parameters are 0.72 to 0.76.

4. How many studies. 0.25 against 0.40 at 5% each way needs 103 studies, a
   cut at 0.320, 0.35G gradients under unit mass at 3.35M a study (NUTS
   4.25M). The measured 0.223 against 0.481 needs only 34 (36 at the
   measured ratios), so 40 was enough to tell these two apart, and 1 of 32
   unit sets and none of the NUTS sets fell on the wrong side of 0.337. But
   0.185 against 0.25, nominal against the line, needs 433 studies and 1.45G.

5. The old readings. Of 14 one-set verdicts from the trajectory and mass
   matrix projects, 5 survive, and every one is a NUTS failure at 0.42 or
   more. No pass survives: a pass needs 0.14 or under on 40 studies to be
   5% from the line, and the best was 0.17 (P 0.12). The two 0.28 fails are
   P 0.33 and do not survive either. Every "on the line" verdict of the last
   two projects was a coin.

What this leaves for day 4: the pooled and prior-variance masses have to be
read on 103 studies or more to call a pass, and log tau's extra spread under
NUTS means its share there needs more than the binomial says.

NumPy only. Fixed seeds. About 2470 seconds.
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day2_share_of_a_real_sampler as s2  # noqa: E402  (the two samplers, read, dispersion; puts the mass matrix project on the path)
from day2_share_of_a_real_sampler import m1, m2, d1, d2, es, KEYS, PARAMS, draw_studies, binom_sd  # noqa: E402

# Every module above reseeds on import; reseed again so these sets of studies are fresh.
es.RNG = np.random.default_rng(191)
d1.RNG = np.random.default_rng(192)
d2.RNG = np.random.default_rng(193)
m2.RNG = np.random.default_rng(194)

L, SIZES = d1.L, d1.SIZES
STUDIES, PASS, BURN = m1.STUDIES, m1.PASS, m1.BURN
SETS_UNIT = 32                   # sets of 40 for unit mass, pooled two and four at a time into 16 of 80 and 8 of 160
SETS_NUTS = 4                    # sets of 40 for NUTS, pooled into 2 of 80 and 1 of 160; a NUTS set is the expensive one
GROUPS = (1, 2, 4)               # sets of 40 per pooled set
Z95 = 1.645                      # one-sided 5%, for both kinds of mistake at once
WARM_UNIT = m1.FAST * m1.WARM_STEPS
WARM_NUTS = (m1.FAST + sum(m1.WINDOWS)) * m1.WARM_STEPS

# Joint band shares at 500 that earlier verdicts were read from, each on one set of 40 studies.
OLD = [
    ("trajectory day 1, unit mass 4 units", 0.20, "pass"),
    ("trajectory day 1, unit mass 14 units", 0.25, "pass"),
    ("trajectory day 1, unit mass 8 units", 0.28, "fail"),
    ("mass matrix day 1, diagonal 2 units", 0.25, "pass"),
    ("mass matrix day 1, diagonal 4.7 units", 0.28, "fail"),
    ("mass matrix day 2, NUTS adapted", 0.60, "fail"),
    ("mass matrix day 3, NUTS diagonal", 0.42, "fail"),
    ("mass matrix day 3, NUTS dense", 0.50, "fail"),
    ("mass matrix day 4, 400 window, set 1", 0.23, "pass"),
    ("mass matrix day 4, 400 window, set 2", 0.23, "pass"),
    ("mass matrix day 4, unit 4 units, set 1", 0.25, "pass"),
    ("mass matrix day 4, unit 4 units, set 2", 0.17, "pass"),
    ("mass matrix day 4, NUTS 100 window, set 1", 0.47, "fail"),
    ("mass matrix day 4, NUTS 100 window, set 2", 0.42, "fail"),
]


def pool(rows, g):
    """Shares of consecutive groups of g sets of 40, which is the share of one set of 40 g studies."""
    x = np.asarray(rows)
    k = len(x) // g
    return x[:k * g].reshape(k, g, *x.shape[1:]).mean(1)


def studies_needed(pa, pb, phia=1.0, phib=1.0):
    """Smallest set of studies at which a cut between pa and pb misreads either one at most 5% of the time.

    With the share's variance phi p (1 - p) / n, the cut c sits where both
    tails are equal, so (pb - pa) = Z95 (sa + sb) / sqrt(n) and n follows.
    """
    sa, sb = np.sqrt(phia * pa * (1 - pa)), np.sqrt(phib * pb * (1 - pb))
    n = (Z95 * (sa + sb) / (pb - pa)) ** 2
    cut = pa + (pb - pa) * sa / (sa + sb)
    return int(np.ceil(n)), cut


def _tail(z):
    """Upper normal tail without SciPy, from the standard library's erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2))


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

    print(f"how many studies: non-centred eight schools at eps {m1.EPS}, sets of {STUDIES} x {nmax} simulations, "
          f"pooled into {', '.join(str(g * STUDIES) for g in GROUPS)}; band on {' / '.join(PARAMS)}; a pass is a joint "
          f"share at or under {PASS}")

    joint = {"unit 4.0": [], "nuts 100": []}
    alone = {"unit 4.0": [], "nuts 100": []}
    steps_nuts = []
    print(f"\n1. unit mass at {s2.UNIT_STEPS * m1.EPS:.1f} units after the fast window, {SETS_UNIT} sets of {STUDIES}")
    for rep in range(SETS_UNIT):
        truth, y = draw_studies(r)
        es.Y = y
        res = s2.read(s2.unit_set(truth), stats, crits)
        joint["unit 4.0"].append(res[nmax][0])
        alone["unit 4.0"].append(res[nmax][1])
        print(f"  set {rep + 1:2d}  joint {res[nmax][0]:.3f}  alone " + " ".join(f"{a:.3f}" for a in res[nmax][1])
              + f"  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n2. NUTS under the 100 window mass, {SETS_NUTS} sets of {STUDIES}")
    for rep in range(SETS_NUTS):
        truth, y = draw_studies(r)
        es.Y = y
        below, st = s2.nuts_set(truth)
        res = s2.read(below, stats, crits)
        joint["nuts 100"].append(res[nmax][0])
        alone["nuts 100"].append(res[nmax][1])
        steps_nuts.append(st)
        print(f"  set {rep + 1:2d}  joint {res[nmax][0]:.3f}  alone " + " ".join(f"{a:.3f}" for a in res[nmax][1])
              + f"  steps {st:4.1f}  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n3. spread against the number of studies at n = {nmax}: sets, mean, sd over sets against the binomial "
          f"at that mean, variance ratio, sd x sqrt(studies), sets on the far side of {PASS}")
    phi = {}
    for name in joint:
        for g in GROUPS:
            x = pool(joint[name], g)
            k, m = len(x), x.mean()
            far = (x > PASS).mean() if name.startswith("unit") else (x <= PASS).mean()
            if k < 2:
                print(f"  {name}  {g * STUDIES:3d} studies  1 set   joint {m:.3f}  far side {far:.2f}")
                continue
            ratio = x.var(ddof=1) / binom_sd(m, g * STUDIES) ** 2
            if g == 1:
                phi[name] = ratio
            print(f"  {name}  {g * STUDIES:3d} studies  {k:2d} sets  joint {m:.3f}  sd {x.std(ddof=1):.3f} "
                  f"({binom_sd(m, g * STUDIES):.3f})  ratio {ratio:4.2f}  sd*sqrt {x.std(ddof=1) * np.sqrt(g * STUDIES):.3f}  "
                  f"far side {far:.2f}")
        a = np.array(alone[name])
        print(f"    alone at 40: " + "  ".join(
            f"{p} {a[:, j].mean():.3f} ratio {a[:, j].var(ddof=1) / binom_sd(a[:, j].mean()) ** 2:4.2f}"
            for j, p in enumerate(PARAMS)))

    pu, pn = np.mean(joint["unit 4.0"]), np.mean(joint["nuts 100"])
    cost_unit = WARM_UNIT + s2.UNIT_STEPS * (BURN + L)
    cost_nuts = WARM_NUTS + np.mean(steps_nuts) * (BURN + L)
    print(f"\n4. studies to tell two shares apart, misreading either at most 5% of the time, and the gradients a "
          f"study costs ({nmax} simulations): unit {cost_unit * nmax / 1e6:.2f}M, NUTS {cost_nuts * nmax / 1e6:.2f}M")
    for label, a, b, fa, fb in [
        ("0.25 against 0.40, binomial", 0.25, 0.40, 1.0, 1.0),
        (f"measured {pu:.3f} against {pn:.3f}, binomial", pu, pn, 1.0, 1.0),
        (f"measured, each at its own variance ratio ({phi['unit 4.0']:.2f}, {phi['nuts 100']:.2f})",
         pu, pn, phi["unit 4.0"], phi["nuts 100"]),
        ("0.185 (four tests at 5%) against 0.25", 0.185, 0.25, 1.0, 1.0),
    ]:
        n, cut = studies_needed(a, b, fa, fb)
        print(f"  {label:58s} {n:5d} studies, cut {cut:.3f}, unit {n * cost_unit * nmax / 1e9:6.2f}G "
              f"NUTS {n * cost_nuts * nmax / 1e9:6.2f}G gradients")
    for g in GROUPS:
        xu, xn = pool(joint["unit 4.0"], g), pool(joint["nuts 100"], g)
        _, cut = studies_needed(pu, pn, phi["unit 4.0"], phi["nuts 100"])
        print(f"  at {g * STUDIES:3d} studies, cut {cut:.3f}: unit sets read as fail {(xu > cut).mean():.2f} "
              f"of {len(xu)}, NUTS sets read as pass {(xn <= cut).mean():.2f} of {len(xn)}")

    print(f"\n5. the old readings, one set of {STUDIES} each: the chance a sampler exactly on {PASS} reads this far on "
          f"its side, binomial and at unit mass's measured ratio; survives if under 0.05")
    kept = 0
    for label, share, verdict in OLD:
        tb = _tail(abs(share - PASS) / binom_sd(PASS))
        tm = _tail(abs(share - PASS) / (binom_sd(PASS) * np.sqrt(max(phi["unit 4.0"], 1.0))))
        ok = tb < 0.05 and (share > PASS) == (verdict == "fail")
        kept += ok
        print(f"  {label:44s} {share:.2f} {verdict}  P {tb:.3f} / {tm:.3f}  {'survives' if ok else 'does not'}")
    print(f"  {kept} of {len(OLD)} survive")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
