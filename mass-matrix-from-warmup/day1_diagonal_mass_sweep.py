"""Day 1 - A diagonal mass from warmup, and the fixed-length sweep again under it.

The trajectory length project ended on two clocks: under unit mass the etas,
posterior sd near one, turn every six units, and mu, sd near three, every
twenty, so fixed lengths passed at four units and fourteen and nowhere else.
A diagonal mass whose inverse is each coordinate's posterior variance should
make every Gaussian coordinate turn once every 2 pi units. This file runs
Stan's warmup schedule on each chain, a fast window of 75 transitions at unit
mass and slow windows of 25, 50 and 100 with the variance re-estimated and
shrunk toward 1e-3 at the end of each, every transition one unit long, then
the same fixed-length sweep from one to fourteen units at eps 0.1 from the
warmed state, 40 studies of 500, the band, chi-squared, rank mean and rank
variance on mu, log tau, theta_1 and eta_1. Going in: the two clocks become
one, mu and eta_1's lag-1 both follow cos(T), the failures sit at multiples
of pi on every coordinate at once, and the window widens to the stretches
around pi / 2 and 3 pi / 2.

1. The mass finds the ratio and has not finished finding mu. The final
   inverse mass is 9.35 on mu (6.10-13.27 over chains), 0.72 on log tau and
   0.83 on eta_1, and the sd ratio mu / eta_1 is 3.35 against the 20 / 6 =
   3.3 the two clocks implied. But mu's estimate is 2.07 after the first
   slow window, 5.77 after the second and 9.35 after the third, still
   climbing at the end of warmup, while eta's moved 0.64 to 0.83. Right on
   the ratio; the 100-transition window is not long enough for mu.

2. Through the first half turn the clocks are one. mu and eta_1's lag-1 go
   +0.58 / +0.57 at one unit, +0.11 / +0.06 at 1.6 where cos is -0.03,
   -0.76 / -0.87 at 3.1 where it is -1.00. After that they come apart: at
   2 pi eta_1 is back at +0.62 and mu only at +0.31, and mu's histogram is
   flat by ten units where eta's is a U again from twelve. Per-chain masses
   spread mu's frequency over chains, so mu dephases and eta, whose mass is
   tight, keeps turning. Half right: one clock for one turn.

3. log tau has no clock. Its lag-1 never goes negative, +0.68 at one unit,
   +0.06 at its lowest at 4.7, back to +0.27 at nine. It is the slow
   coordinate the trajectory length project never saw, because under unit
   mass mu failed first. It is the worst parameter at 1.0 and 1.6 units,
   band 0.97 and 0.60 on it alone, and not predicted: the quarter turn at
   pi / 2, where mu and eta are both near zero lag, fails on log tau.

4. The window narrows, it does not widen. One pass in nineteen lengths:
   2.0 units, band 0.25 on the line. 4.7, three quarters of a turn, is
   0.28 and 5.0 is 0.38, both on mu; six to seven units are 100% on
   eta_1 (U, ends 0.28-0.30) and eight is 0.85, three units is the W again (1.00,
   eta_1 1.20 0.87 ... 1.18 1.13 ... 1.17), and nine to fourteen fail on
   eta's second turn. The unit-mass sweep had two passes, at four and
   fourteen. Wrong: the mass moved the window to two units and put
   log tau under it.

5. The rank mean passes every length again, 0.05 to 0.15 at 500, and the
   acceptance is 0.996 at every length as it was under unit mass. Neither
   says anything about the mass.

6. Cost. The pass at two units samples at 20 x 99 = 1980 gradients a
   simulation, half of unit mass's 3960 at four units, but the warmup is
   2500 and the burn at the swept length 1000 more, 5480 in all. Per
   simulation at L = 99 the mass costs more than it saves; it pays only if
   the warmup is shared over more draws, which day 4 measures. One right,
   one half, one wrong, and log tau and the cost not predicted.

NumPy only. Fixed seeds. About 880 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "trajectory-length-found-blind"))
import day1_period_sweep as d0  # noqa: E402  (puts the eight schools and rank-test modules on the path)
from day1_period_sweep import d1, d2, es, ends, per_param, shape, KEYS, PARAMS, draw_studies, params  # noqa: E402

# The trajectory length modules reseed the imported generators when they load;
# reseed again so these studies share nothing with that project's.
es.RNG = np.random.default_rng(121)
d1.RNG = np.random.default_rng(122)
d2.RNG = np.random.default_rng(123)
RNG = np.random.default_rng(124)

L, SIZES = d1.L, d1.SIZES
EPS = d0.EPS                     # 0.1, as in the unit-mass sweep, so acceptance is not what changes
STUDIES = d0.STUDIES
LAG_CHAINS = d0.LAG_CHAINS
PASS = d0.PASS
WARM_STEPS = 10                  # every warmup transition is one unit long
FAST = 75                        # Stan's initial fast window, unit mass, nothing estimated
WINDOWS = (25, 50, 100)          # the slow windows, doubling; the variance is re-estimated at the end of each
BURN = 50                        # transitions at the swept length, from the warmed state, before the ranks start
# one to fourteen units, plus the quarter, half, three-quarter and full turn of a unit clock
STEPS = sorted(set(range(10, 141, 10)) | {16, 31, 47, 63, 79})
NAMES = ["mu", "log tau"] + [f"eta_{j + 1}" for j in range(8)]


def hmc_mass(z, steps, minv):
    """Non-centred HMC with a per-chain diagonal mass; minv is the inverse mass, (r, 10).

    Momentum is drawn N(0, M), so p = g / sqrt(minv), the kinetic energy is
    0.5 sum minv p^2 and the position moves by eps minv p. A coordinate with
    posterior variance s^2 and minv = s^2 then turns once every 2 pi units,
    whatever s is, which is the point of the mass.
    """
    pot, grad = es.MODELS["noncentred"]
    p = es.RNG.standard_normal(z.shape) / np.sqrt(minv)
    h0 = pot(z) + 0.5 * (minv * p * p).sum(1)
    zn = z.copy()
    g = grad(zn)
    for _ in range(steps):
        p = p - 0.5 * EPS * g
        zn = zn + EPS * minv * p
        g = grad(zn)
        p = p - 0.5 * EPS * g
    dh = pot(zn) + 0.5 * (minv * p * p).sum(1) - h0
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(es.RNG.random(z.shape[0])) < -np.nan_to_num(dh, nan=np.inf))
    return np.where(acc[:, None], zn, z), acc


def regularise(var, n):
    """Stan's shrinkage of a window's variance toward 1e-3, weight 5 / (n + 5)."""
    return (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))


def warmup(r):
    """Fast window at unit mass, then the doubling slow windows, each chain on its own variance.

    Returns the state, the final inverse mass (r, 10) and the inverse mass after
    each slow window, so how fast the estimate settles can be read off.
    """
    z = es.prior_start("noncentred", r)
    minv = np.ones((r, 10))
    for _ in range(FAST):
        z, _ = hmc_mass(z, WARM_STEPS, minv)
    history = []
    for n in WINDOWS:
        s1 = np.zeros((r, 10))
        s2 = np.zeros((r, 10))
        for _ in range(n):
            z, _ = hmc_mass(z, WARM_STEPS, minv)
            s1 += z
            s2 += z * z
        m = s1 / n
        var = s2 / n - m * m
        minv = regularise(np.maximum(var, 0.0), n)
        history.append(minv.copy())
    return z, minv, history


def lag1(kept):
    """Lag-1 autocorrelation of each pilot, each chain centred on its own mean, pooled over chains."""
    x = kept - kept.mean(0)
    return (x[1:] * x[:-1]).sum((0, 1)) / (x * x).sum((0, 1))


def sbc_mass(z, truth, minv, steps):
    """BURN transitions at the swept length from the warmed state, then L with the ranks accumulated."""
    r = z.shape[0]
    for _ in range(BURN):
        z, _ = hmc_mass(z, steps, minv)
    below = np.zeros((r, 4), dtype=np.int32)
    acc = 0.0
    kept = np.empty((L, LAG_CHAINS, 3))
    for k in range(L):
        z, a = hmc_mass(z, steps, minv)
        acc += a.mean()
        p = params("noncentred", z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, acc / L, kept


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax
    warm_cost = (FAST + sum(WINDOWS)) * WARM_STEPS

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    truth, y = draw_studies(r)
    es.Y = y
    print(f"mass from warmup, day 1: non-centred hmc at eps {EPS}, {STUDIES} x {nmax} studies, L = {L}, "
          f"warmup {FAST} + {' + '.join(map(str, WINDOWS))} transitions at {WARM_STEPS * EPS:.0f} unit "
          f"({warm_cost} gradients), burn {BURN} at each length")

    z, minv, history = warmup(r)
    print(f"\n1. the inverse mass, median over chains (and 10-90%), after each slow window, against "
          f"unit mass's 1  ({time.time() - t0:.0f} s)")
    for n, h in zip(WINDOWS, history):
        print(f"  after {n:3d}  " + "  ".join(
            f"{NAMES[j]} {np.median(h[:, j]):.2f} ({np.quantile(h[:, j], 0.1):.2f}-{np.quantile(h[:, j], 0.9):.2f})"
            for j in (0, 1, 2, 3)))
    sd = np.sqrt(minv)
    print(f"  final sd ratio mu / eta_1, median {np.median(sd[:, 0] / sd[:, 2]):.2f}, "
          f"the unit-mass clocks' ratio was about 20 / 6 = 3.3")

    runs = {}
    print("\n2. acceptance and lag-1 autocorrelation of mu / log tau / eta_1 under the adapted mass, "
          "against cos(T), now the prediction for all three")
    for steps in STEPS:
        below, acc, kept = sbc_mass(z.copy(), truth, minv, steps)
        units = steps * EPS
        runs[steps] = below.reshape(STUDIES, nmax, 4)
        rho = lag1(kept)
        print(f"  T = {units:4.1f} ({steps:3d} steps)  acc {acc:.3f}  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / "
              f"{rho[2]:+.2f}  cos(T) {np.cos(units):+.2f}  {time.time() - t0:5.0f} s", flush=True)

    print(f"\n3. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for steps, rr in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  T = {steps * EPS:4.1f}  {cells}")

    print(f"\n4. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for steps, rr in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  T = {steps * EPS:4.1f}  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n5. pooled rank histograms of eta_1 and mu in tenths at n = {nmax}")
    for steps, rr in runs.items():
        print(f"  T = {steps * EPS:4.1f}  eta_1 {shape(rr, 3)}   mu {shape(rr, 0)}")

    print(f"\n6. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}, "
          f"gradients a simulation = warmup {warm_cost} + steps x ({BURN} + {L})")
    for steps, rr in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        units = steps * EPS
        print(f"  T = {units:4.1f}  T / 2pi {units / (2 * np.pi):.2f}  band {share:.2f}  "
              f"{'pass' if share <= PASS else 'fail'}  worst {worst:8s} "
              f"{warm_cost + steps * (BURN + L):6d} gradients")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
