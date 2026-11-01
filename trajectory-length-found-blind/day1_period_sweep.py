"""Day 1 - The period guess: non-centred HMC at eps 0.1 over trajectory lengths from 1 to 14 units.

Eight schools day 4 ended with a guess: the non-centred sampler's break at
eps 0.3 x 20 steps, six units, was on eta_1 with a deep U and not on mu or
a rejection, and eta has unit mass and a near unit-normal posterior, so its
trajectories turn once every 2 pi units and six units bring it nearly back
to its start. This file tests the guess by holding eps at 0.1, where
acceptance is 0.996 at every length and nothing is a rejection, and sweeping
the step count from 10 to 140, one to fourteen units, with 2 pi and 4 pi
added to the nearest step. At each length the lag-1 autocorrelation of mu,
log tau and eta_1 is read against cos(T), the prediction for a unit normal
under unit mass, and 40 studies of 500 simulations go under the band, the
chi-squared, the rank mean and the rank variance on mu, log tau, theta_1
and eta_1. Going in: eta_1's failures sit at multiples of 2 pi, the lengths
between them pass, six units is the first, and the rank mean sees none of
it.

1. The six-unit break is eta's period. eps 0.1 x 60 fails the band on eta_1
   on 100% of studies at 500 with ends 2.43 and 2.40 in tenths, which is
   eps 0.3 x 20's 2.45 and 2.56 from day 4 at a third of the step size, so
   it was the length. eta_1's lag-1 autocorrelation follows cos(T) to a few
   hundredths through the first turn, +0.50 at one unit, -0.95 at three
   where cos is -0.99, +0.85 at 2 pi, -0.86 at nine, and it is the one
   parameter that does. Right.

2. The failures sit at multiples of pi, not 2 pi, and come in two shapes.
   At 2 pi and 4 pi the chain returns, rho1 +0.85 and +0.67, and the ranks
   are a U, ends 0.48 and 0.25 at six and 12.6 units. At pi and 3 pi it
   reflects, rho1 -0.95 at three units and -0.86 at nine, and the ranks are
   a W, 1.44 and 1.48 at the ends with 1.45 and 1.43 in the middle tenths
   at three units; the band rejects eta_1 on 100% of studies at both. The
   rank tests project measured HMC antithetic at lag 1 and called it
   harmless; at a full reflection it is a failure the band catches as fast
   as the U. Half right: the period is the right clock, the guess counted
   half its ticks.

3. The ranks put eta's period at about six units, not 6.28. The U is
   deeper at 6.0 than at 6.3, ends 0.48 against 0.35 and 2.43 against 1.69
   in the top tenth, and again at 12.0 against 12.6, 0.36 against 0.25, so
   the eta_1 posterior is a little narrower than its prior and the
   trajectory turns a touch early. Not predicted; a 4% correction the rank
   histograms resolve and the lag-1 numbers, 0.84 against 0.85, do not.

4. The lengths between the ticks do not pass, because mu has its own clock.
   mu's rho1 falls from +0.92 at one unit through zero near five to -0.96
   at ten and is on its way back at fourteen, -0.48, a period near twenty
   units from a posterior sd near three. Too short it is a U, ends 0.39 at
   one unit and 0.25 at two, the day 3 failure; at ten and eleven units it
   is the W, 1.36 and 1.33 at the ends with 1.39 twice in the middle, band
   100% on mu where eta_1 passes. The two clocks together leave two passes
   in fourteen: four units, band 0.20, and fourteen, 0.25 on the line, with
   eight at 0.28 just over it and five and seven at 0.40 and 0.55 off
   eta's shoulders. Wrong.

5. The rank mean passes every length, 0.07 to 0.28, at every size, as it
   passed the funnel's U, because the U and the W are both even in the
   rank. The chi-squared and the rank variance follow the band and reach
   100% later; at six units the band is at 0.93 at 20 simulations where the
   chi-squared is at 0.57. Right, and the one thing a blind rule gets for
   free: a shift test cannot tell a tuner anything about its length.

6. Cost. The cheapest pass is the four-unit one at 3960 gradients a
   simulation, the day 4 price; fourteen units costs 13860 for a worse
   pass, and every length from five to thirteen is paid for and fails.
   A rule that lands on the window has to find a one-unit-wide pass between
   two periods it cannot see in the acceptance rate, which is 0.996
   everywhere. Day 2 asks whether jittering the step count fills the ticks
   in. Two right, one half, one wrong.

NumPy only. Fixed seeds. About 1400 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ES = os.path.join(HERE, "..", "eight-schools-sbc-with-data")
sys.path.insert(0, ES)
sys.path.insert(0, os.path.join(HERE, "..", "rank-tests-sharper-sbc"))
sys.path.insert(0, os.path.join(HERE, "..", "sampler-correctness-sbc"))
import day1_centred_vs_noncentred as es  # noqa: E402
import day1_ecdf_bands as d1  # noqa: E402
import day2_shape_tests as d2  # noqa: E402
from day2_sbc_with_data import per_param, shape  # noqa: E402
from day3_eta_ranks_and_divergences import KEYS, PARAMS, draw_studies, params  # noqa: E402
from day4_step_size_and_the_neck import ends  # noqa: E402

# Eight schools day 4 reseeds the imported generators when it loads; reseed
# again so these studies share nothing with that project's.
es.RNG = np.random.default_rng(81)
d1.RNG = np.random.default_rng(82)
d2.RNG = np.random.default_rng(83)
RNG = np.random.default_rng(84)

L, SIZES = d1.L, d1.SIZES
EPS = 0.1
BURN = 100
STUDIES = 40
LAG_CHAINS = 400                 # chains whose mu, log tau and eta_1 pilots are kept for the lag-1 correlations
# step counts: one to fourteen units, plus 2 pi and 4 pi to the nearest step
STEPS = sorted(set(range(10, 141, 10)) | {63, 126})
PASS = 0.25                      # the nominal 0.19 of four tests at 5% plus the error of 40 studies


def sbc_pilot(truth, y, steps):
    """Burn, then L transitions at thin 1 with the ranks of the truth accumulated.

    Returns the ranks (r, 4), the mean acceptance and the pilots of mu, log tau
    and eta_1 on the first LAG_CHAINS chains, (L, LAG_CHAINS, 3).
    """
    es.Y = y
    es.STEPS = steps
    r = y.shape[0]
    z = es.prior_start("noncentred", r)
    for _ in range(BURN):
        z, _, _ = es.hmc("noncentred", z, EPS)
    below = np.zeros((r, 4), dtype=np.int32)
    acc = 0.0
    kept = np.empty((L, LAG_CHAINS, 3))
    for k in range(L):
        z, a, _ = es.hmc("noncentred", z, EPS)
        acc += a.mean()
        p = params("noncentred", z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, acc / L, kept


def lag1(kept):
    """Lag-1 autocorrelation of each pilot, each chain centred on its own mean, pooled over chains."""
    x = kept - kept.mean(0)
    return (x[1:] * x[:-1]).sum((0, 1)) / (x * x).sum((0, 1))


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
    print(f"trajectory length, the period guess: non-centred hmc at eps {EPS}, {STUDIES} x {nmax} studies, "
          f"burn {BURN}, L = {L}")
    print(f"  the prior studies: tau < {es.NECK:.0f} on {100 * np.mean(np.exp(truth[:, 1]) < es.NECK):.1f}%, "
          f"median tau {np.median(np.exp(truth[:, 1])):.2f}")

    runs = {}
    print("\n1. acceptance and lag-1 autocorrelation of mu / log tau / eta_1 against cos(T), "
          "the unit-normal prediction for eta")
    for steps in STEPS:
        below, acc, kept = sbc_pilot(truth, y, steps)
        units = steps * EPS
        runs[steps] = below.reshape(STUDIES, nmax, 4)
        rho = lag1(kept)
        print(f"  T = {units:4.1f} ({steps:3d} steps)  acc {acc:.3f}  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / "
              f"{rho[2]:+.2f}  cos(T) {np.cos(units):+.2f}  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n2. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for steps, rr in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  T = {steps * EPS:4.1f}  {cells}")

    print(f"\n3. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for steps, rr in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  T = {steps * EPS:4.1f}  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n4. pooled rank histograms of eta_1 and mu in tenths at n = {nmax}")
    for steps, rr in runs.items():
        print(f"  T = {steps * EPS:4.1f}  eta_1 {shape(rr, 3)}   mu {shape(rr, 0)}")

    print(f"\n5. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}, "
          f"gradients a simulation = steps x {L}")
    for steps, rr in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        units = steps * EPS
        print(f"  T = {units:4.1f}  T / 2pi {units / (2 * np.pi):.2f}  band {share:.2f}  "
              f"{'pass' if share <= PASS else 'fail'}  worst {worst:8s} {steps * L:6d} gradients")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
