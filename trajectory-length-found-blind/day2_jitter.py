"""Day 2 - Jitter: the step count drawn uniformly around its centre each transition.

Day 1 swept non-centred HMC at eps 0.1 from one to fourteen units and found
two clocks: eta_1 fails at every multiple of pi, a U where the chain returns
(six units, 12.6) and a W where it reflects (three, nine), and mu fails on
its own period near twenty units, a U under two units and a W at ten and
eleven. Two lengths in fourteen passed, four units and fourteen. The usual
fix for a resonance is to stop using one length: draw the step count fresh
for every chain at every transition, so no chain sits on a period for long.
This file keeps day 1's eps, studies, burn and tests and replaces the fixed
count with a uniform draw on [c(1 - w), c(1 + w)] for centres at three, four,
six and ten units (eta's reflection, the window, eta's return, mu's W) and
widths w = 0.5 and 1, where w = 1 draws from one step to twice the centre.
The mean step count is the centre, so the cost in gradients a simulation is
day 1's at the same centre. Going in: jitter fills in eta's ticks at three
and six units because a spread of +/-1.5 units covers half a period of a
six-unit clock, it does not fix mu's W at ten units because +/-5 units is a
quarter of a twenty-unit clock, and it costs the four-unit window little.

1. Jitter fills in eta's ticks. At three units, a reflection with rho1
   -0.95 and a W on day 1, the band rejects eta_1 alone on 12% of studies
   at 500 with +/-50% and 7% with +/-100%, against 100% fixed, and its
   histogram is flat to 0.04 in every tenth. At six units, day 1's deep U
   with ends 0.48, eta_1 is at 12% and 7%, ends 0.20. Right.

2. It fixes mu's W at ten units too. +/-5 units is a quarter of mu's
   twenty-unit clock and the guess was that a quarter would not be enough;
   mu's band at 500 is 10% at +/-50% and 7% at +/-100%, against 100% fixed,
   ends 0.20. Wrong. The lag-1 numbers say why it did not need to be:
   +/-50% only halves the antithetic correlation, mu's rho1 -0.63 against
   -0.96, eta_1's -0.59 at three units against -0.95, and that is already
   below what the band sees at 500 simulations. +/-100% takes both to
   within 0.05 of zero. Day 1's failures needed |rho1| near one.

3. What jitter does not fix is mu at three units, which is not a resonance
   but a trajectory too short for mu's posterior: rho1 +0.58 and +0.51,
   band on mu 20% and 23%, ends of mu's histogram 1.04 to 1.07 in tenths,
   a faint version of day 1's short-length U. The joint band is 0.42 and
   0.40, the only cells clearly over the line.

4. Everything from four to ten units now sits on the line. The joint band
   at 500 is 0.23 to 0.40, with no single parameter over 0.17 and every
   end share 0.20 to 0.21. Forty studies put the error of a share near
   0.25 at about 0.07, so the pass/fail column, six units +/-50% and ten
   units +/-100% pass at 0.23 and four units +/-100% fails at 0.40, does
   not separate these cells; the per-parameter numbers say none of them is
   a failure of the day 1 kind. Fixed four units on day 1 was 0.20.

5. Cost. The mean step count is the centre, so a simulation costs what day
   1's fixed length did at the same centre: 3960 gradients at four units,
   5940 at six. The vectorised run here integrates every chain to the
   longest draw, so it paid 1.5x and 2x that in wall time; a per-chain
   sampler pays the mean. Little cost to the window: right, with the
   caveat in 4 that the window's own pass is within the noise of its
   neighbours.

The result for a blind rule: fixed lengths gave it a one-unit window in
fourteen, jittered lengths give it a plateau from four to ten units where
the only thing left to choose is cost, and the acceptance rate, 0.996 in
every cell, still says nothing. Day 3 builds NUTS from scratch to see
whether the U-turn criterion lands on the plateau without being told the
period. Two right, one wrong, one not predicted.

NumPy only. Fixed seeds. About 870 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_period_sweep as d0  # noqa: E402  (puts the eight schools and rank-test modules on the path)
from day1_period_sweep import d1, d2, es, ends, per_param, shape, KEYS, PARAMS, draw_studies, params  # noqa: E402

# Day 1 reseeds on import; reseed again so the jittered studies are fresh draws.
es.RNG = np.random.default_rng(91)
d1.RNG = np.random.default_rng(92)
d2.RNG = np.random.default_rng(93)
RNG = np.random.default_rng(94)

L, SIZES = d1.L, d1.SIZES
EPS, BURN, STUDIES, PASS = d0.EPS, d0.BURN, d0.STUDIES, d0.PASS
LAG_CHAINS = d0.LAG_CHAINS
CENTRES = [30, 40, 60, 100]      # three, four, six and ten units at eps 0.1
WIDTHS = [0.5, 1.0]


def draw_steps(centre, width, chains):
    """Per-chain step counts, uniform on the integers in [centre (1 - width), centre (1 + width)], at least 1."""
    lo = max(1, int(round(centre * (1 - width))))
    hi = int(round(centre * (1 + width)))
    return RNG.integers(lo, hi + 1, chains)


def hmc_jitter(z, centre, width):
    """Unit-mass non-centred HMC where every chain runs its own step count this transition.

    The whole array is integrated to the longest count and each chain's state is
    frozen once its own count is reached, so a chain's trajectory does not depend
    on any other chain's. Returns (z, accepted, steps drawn).
    """
    pot, grad = es.MODELS["noncentred"]
    n = draw_steps(centre, width, z.shape[0])
    p = es.RNG.standard_normal(z.shape)
    h0 = pot(z) + 0.5 * (p * p).sum(1)
    zn = z.copy()
    g = grad(zn)
    for s in range(n.max()):
        live = (s < n)[:, None]
        p1 = p - 0.5 * EPS * g
        z1 = zn + EPS * p1
        g1 = grad(z1)
        p1 = p1 - 0.5 * EPS * g1
        zn = np.where(live, z1, zn)
        p = np.where(live, p1, p)
        g = np.where(live, g1, g)
    dh = pot(zn) + 0.5 * (p * p).sum(1) - h0
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(es.RNG.random(z.shape[0])) < -np.nan_to_num(dh, nan=np.inf))
    return np.where(acc[:, None], zn, z), acc, n


def sbc_pilot(truth, y, centre, width):
    """Day 1's pilot with the jittered transition: ranks (r, 4), acceptance, kept pilots, mean steps."""
    es.Y = y
    r = y.shape[0]
    z = es.prior_start("noncentred", r)
    for _ in range(BURN):
        z, _, _ = hmc_jitter(z, centre, width)
    below = np.zeros((r, 4), dtype=np.int32)
    acc, steps = 0.0, 0.0
    kept = np.empty((L, LAG_CHAINS, 3))
    for k in range(L):
        z, a, n = hmc_jitter(z, centre, width)
        acc += a.mean()
        steps += n.mean()
        p = params("noncentred", z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, acc / L, kept, steps / L


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

    truth, y = draw_studies(r)
    print(f"trajectory length, jitter: non-centred hmc at eps {EPS}, {STUDIES} x {nmax} studies, burn {BURN}, L = {L}")

    runs = {}
    print("\n1. acceptance, mean steps and lag-1 autocorrelation of mu / log tau / eta_1")
    for c in CENTRES:
        for w in WIDTHS:
            below, acc, kept, mean_steps = sbc_pilot(truth, y, c, w)
            runs[c, w] = below.reshape(STUDIES, nmax, 4)
            rho = d0.lag1(kept)
            print(f"  T = {c * EPS:4.1f} +/- {100 * w:3.0f}%  acc {acc:.3f}  steps {mean_steps:5.1f}  "
                  f"rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f}  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n2. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for (c, w), rr in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  T = {c * EPS:4.1f} +/- {100 * w:3.0f}%  {cells}")

    print(f"\n3. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for (c, w), rr in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  T = {c * EPS:4.1f} +/- {100 * w:3.0f}%  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n4. pooled rank histograms of eta_1 and mu in tenths at n = {nmax}")
    for (c, w), rr in runs.items():
        print(f"  T = {c * EPS:4.1f} +/- {100 * w:3.0f}%  eta_1 {shape(rr, 3)}   mu {shape(rr, 0)}")

    print(f"\n5. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}, "
          f"gradients a simulation = mean steps x {L}")
    for (c, w), rr in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        print(f"  T = {c * EPS:4.1f} +/- {100 * w:3.0f}%  band {share:.2f}  {'pass' if share <= PASS else 'fail'}  "
              f"worst {worst:8s} {c * L:6d} gradients")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
