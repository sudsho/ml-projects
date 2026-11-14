"""Day 4 - What the warmup costs.

Day 1 passed once, fixed length at two units under the mass from Stan's
three windows, band 0.25 on the line, and said the mass would pay only if
the warmup were shared over more draws. Day 3 found 40 studies move a share
by 0.2. This file keeps the state and the mass at the end of every window,
day 1's three and two more doublings to 200 and 400, and runs the fixed
length at two and four units from each, on two independent sets of 40
studies of 500, then day 2's tree after the 100 window and after the 400.
Written before the run: two units passes from the 100 window on and more
easily after 200 and 400, mu's inverse mass levels off near 12, and the tree
still fails on log tau under the longest warmup.

1. mu settles and log tau does not. mu's inverse mass is 2.05, 5.73, 9.36,
   10.43, 10.76 over the five windows, the same in both sets to 0.03, and
   its spread over chains (sd of the log) falls 0.70 to 0.12. So it levels
   off near 10.8, not 12. log tau's goes 0.50, 0.55, 0.72, 0.90, 1.02 and
   is still climbing at 400 with a spread of 0.44, chains from 0.65 to
   1.67. Half right on mu; log tau was not predicted.

2. Day 1's pass was the draw. From the 100 window two units is 0.47 and
   0.38 on the two sets, worst on log tau, 0.33 and 0.25 alone. The same
   sampler read 0.25 on day 1's studies. Wrong, and it is day 3's 0.2
   again.

3. Two units passes only after the 400 window, 0.23 on both sets. After 200
   it is 0.42 and 0.42. What moves is log tau, its band alone 0.60 / 0.42
   after 25, 0.35 / 0.20 after 200, 0.12 / 0.05 after 400, and its lag-1
   +0.37 to +0.15. mu is done by the 100 window, 0.07 and 0.03 alone. So
   the warmup's length is set by the coordinate with no clock, and the
   doubling that fixes it costs 4000 gradients on its own.

4. Four units under a short warmup is worse than no mass. Unit mass at four
   units passes on both sets, 0.25 and 0.17. After the 25 window it is
   1.00 and 0.97, all of it eta_1, lag-1 -0.77: a noisy mass puts eta on
   its half turn. It is 0.88 / 0.82 after 100 and 0.30 / 0.28 after 400,
   not a pass anywhere under the mass.

5. The tree does not need the long warmup and does not pass with it. 40.3
   steps and band 0.47 after the 100 window, 35.0 steps and 0.42 after
   the 400, log tau alone 0.38 and 0.33, lag-1 +0.40 and +0.34. Right. The
   better mass shortens the tree by five steps and leaves log tau where
   it was, so the fixed two units passes where the tree fails because it
   is shorter on log tau's side, not because of the mass.

6. Cost. The one pass under the mass is 8500 of warmup plus 20 x 149, 11480
   gradients a simulation, against 6710 for unit mass at four units with
   its fast window, the tuned and jittered length's 4500 and the tree's
   13711 for a fail. At 20 steps a draw against 40 the mass is ahead once
   a chain keeps more than 338 draws; at 1000 draws it is 2920 gradients
   per 99 against 4232. At SBC's 99 draws a chain it never pays. One
   right, one half, one wrong.

NumPy only. Fixed seeds. About 1300 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_diagonal_mass_sweep as m1  # noqa: E402  (the windows; it puts the older projects on the path)
import day2_nuts_under_the_mass as m2  # noqa: E402  (the tree under a diagonal mass)
from day1_diagonal_mass_sweep import d1, d2, es, ends, per_param, KEYS, PARAMS, draw_studies  # noqa: E402

# Days 1 and 2 reseed on import; reseed again so these studies are fresh draws.
es.RNG = np.random.default_rng(151)
d1.RNG = np.random.default_rng(152)
d2.RNG = np.random.default_rng(153)
m2.RNG = np.random.default_rng(154)

L, SIZES = d1.L, d1.SIZES
EPS = m1.EPS
STUDIES, PASS, BURN = m1.STUDIES, m1.PASS, m1.BURN
FAST, WARM_STEPS = m1.FAST, m1.WARM_STEPS
WINDOWS = (25, 50, 100, 200, 400)    # day 1's three slow windows and two more doublings
LENGTHS = (20, 40)                   # two units, day 1's one pass, and four, unit mass's
REPS = 2                             # independent sets of 40 studies, since 40 move a share by 0.2
NUTS_AFTER = (3, 5)                  # the tree after day 1's schedule and after the longest, first set only


def warmup_snapshots(r):
    """Day 1's warmup with the state and the inverse mass kept at the end of every window.

    The snapshot after k slow windows is what a warmup that stopped there
    would have handed over, so one pass gives every schedule. Snapshot 0 is
    the fast window alone, unit mass. Returns a list of (z, minv, gradients).
    """
    z = es.prior_start("noncentred", r)
    minv = np.ones((r, 10))
    for _ in range(FAST):
        z, _ = m1.hmc_mass(z, WARM_STEPS, minv)
    snaps = [(z.copy(), minv.copy(), FAST * WARM_STEPS)]
    done = FAST
    for n in WINDOWS:
        s1 = np.zeros((r, 10))
        s2 = np.zeros((r, 10))
        for _ in range(n):
            z, _ = m1.hmc_mass(z, WARM_STEPS, minv)
            s1 += z
            s2 += z * z
        m = s1 / n
        minv = m1.regularise(np.maximum(s2 / n - m * m, 0.0), n)
        done += n
        snaps.append((z.copy(), minv.copy(), done * WARM_STEPS))
    return snaps


def spread(minv, j):
    """Median over chains and the 10-90% range of one coordinate's inverse mass."""
    return f"{np.median(minv[:, j]):5.2f} ({np.quantile(minv[:, j], 0.1):5.2f}-{np.quantile(minv[:, j], 0.9):5.2f})"


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    print(f"mass from warmup, day 4: non-centred at eps {EPS}, {REPS} sets of {STUDIES} x {nmax} studies, L = {L}, "
          f"fast window {FAST}, slow windows {' + '.join(map(str, WINDOWS))} at {WARM_STEPS * EPS:.0f} unit, "
          f"burn {BURN}; a pass is band at or under {PASS} on the four together")

    table = {}        # (snapshot, steps) -> list over sets of (joint band, per-parameter band, lag-1, ends)
    nuts = {}
    for rep in range(REPS):
        truth, y = draw_studies(r)
        es.Y = y
        snaps = warmup_snapshots(r)
        print(f"\nset {rep + 1}. 1. the inverse mass after each window, median over chains (10-90%), and the "
              f"spread of mu's over chains as sd of its log  ({time.time() - t0:.0f} s)")
        for k, (_, minv, cost) in enumerate(snaps):
            name = "fast only" if k == 0 else f"to {WINDOWS[k - 1]:3d}"
            print(f"  {name:9s} {cost:5d} gradients  mu {spread(minv, 0)}  log tau {spread(minv, 1)}  "
                  f"eta_1 {spread(minv, 2)}  sd log minv mu {np.log(minv[:, 0]).std():.2f} "
                  f"log tau {np.log(minv[:, 1]).std():.2f}")

        print(f"set {rep + 1}. 2. fixed length from each snapshot: acceptance, lag-1 of mu / log tau / eta_1, band on "
              f"the four at n = {nmax}, each parameter alone ({' / '.join(PARAMS)}), end tenths")
        for k, (z, minv, cost) in enumerate(snaps):
            name = "fast only" if k == 0 else f"to {WINDOWS[k - 1]:3d}"
            for steps in LENGTHS:
                below, acc, kept = m1.sbc_mass(z.copy(), truth, minv, steps)
                rr = below.reshape(STUDIES, nmax, 4)
                rho = m1.lag1(kept)
                joint = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
                alone = per_param(rr, stats[nmax], crits[nmax], "band")
                flat = rr.reshape(r, 4)
                table.setdefault((k, steps), []).append((joint, alone))
                print(f"  {name:9s} T = {steps * EPS:3.1f}  acc {acc:.3f}  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / "
                      f"{rho[2]:+.2f}  band {joint:.2f}  alone " + " ".join(f"{a:.2f}" for a in alone)
                      + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4))
                      + f"  {time.time() - t0:5.0f} s", flush=True)

        if rep == 0:
            print(f"set 1. 3. the tree from two snapshots: steps a transition, depth 5 and 6 shares, lag-1, band, "
                  f"each parameter alone, end tenths")
            for k in NUTS_AFTER:
                z, minv, cost = snaps[k]
                below, kept, dshare, steps, div, _ = m2.sbc_nuts(z.copy(), truth, minv, "multinomial")
                rr = below.reshape(STUDIES, nmax, 4)
                rho = m1.lag1(kept)
                joint = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
                alone = per_param(rr, stats[nmax], crits[nmax], "band")
                flat = rr.reshape(r, 4)
                nuts[k] = (joint, alone, steps.mean(), cost)
                print(f"  to {WINDOWS[k - 1]:3d}    steps {steps.mean():5.1f}  depth {dshare[5]:.2f} {dshare[6]:.2f}  "
                      f"div {100 * div.mean():.2f}%  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f}  band {joint:.2f}  "
                      f"alone " + " ".join(f"{a:.2f}" for a in alone)
                      + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4))
                      + f"  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n4. verdict: band on the four in each set and their mean, the worst parameter by its mean band alone, "
          f"gradients a simulation = warmup + steps x ({BURN} + {L}), and the same with the warmup spread over "
          f"1000 draws a chain")
    for (k, steps), rows in table.items():
        name = "fast only" if k == 0 else f"to {WINDOWS[k - 1]:3d}"
        joints = [j for j, _ in rows]
        alone = np.mean([a for _, a in rows], axis=0)
        warm = (FAST + sum(WINDOWS[:k])) * WARM_STEPS
        cost = warm + steps * (BURN + L)
        per99 = (warm + steps * (BURN + 1000)) * 99 / 1000
        verdict = "pass" if max(joints) <= PASS else ("split" if min(joints) <= PASS else "fail")
        print(f"  {name:9s} T = {steps * EPS:3.1f}  band " + " ".join(f"{j:.2f}" for j in joints)
              + f"  mean {np.mean(joints):.2f}  {verdict:5s} worst {PARAMS[int(np.argmax(alone))]:8s} "
              f"({alone.max():.2f})  {cost:5d} gradients, {per99:5.0f} per 99 draws of 1000")
    for k, (joint, alone, steps, warm) in nuts.items():
        print(f"  nuts to {WINDOWS[k - 1]:3d}        band {joint:.2f}             "
              f"{'pass' if joint <= PASS else 'fail'}  worst {PARAMS[int(np.argmax(alone))]:8s} ({max(alone):.2f})  "
              f"{warm + steps * (BURN + L):5.0f} gradients")
    print("  against: the tuned and jittered length's 4500, unit mass swept to four units 3960 sampling and 5960 "
          "with its burn, day 1's two units 5480")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
