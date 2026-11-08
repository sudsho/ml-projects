"""Day 4 - A length tuned on the chain: warmup sets the trajectory from the IACT of the widest parameter.

Day 3's NUTS found the window's length, 4.2 units, and failed in it on mu,
band 0.90 at 500, because the U-turn sum is mostly the eight etas, which
turn at about pi, and mu needs longer. This file makes the opposite choice.
Every chain warms up for 100 transitions at one unit, reads the lag-1
autocorrelation of each of its ten coordinates on the last 80, picks the
widest coordinate (largest warmup variance) or the slowest (largest lag-1),
and sets its trajectory to that coordinate's quarter period: under unit mass
a Gaussian coordinate has lag-1 cos(omega T), so arccos(rho) at one unit
gives omega, and pi / (2 omega) is the length where its lag-1 would be zero.
Reading the IACT as an AR(1), (1 + rho) / (1 - rho), is the same
measurement. Then 50 transitions at the tuned length and day 1's 99 at it,
fixed or jittered +/-50% around it as day 2 taught, on 40 studies of 500.
No prediction written; day 3 left the question of whether a rule that looks
at mu fixes mu.

1. Both rules pick mu and land in the window. Widest picks mu on 99% of
   chains and slowest on 98%, the rest log tau, so they are one rule. mu's
   lag-1 at one unit is +0.92 (median over chains), against +0.52 for log
   tau and +0.49 and +0.46 for eta_1 and eta_2, and arccos(0.92) puts the quarter
   period at 3.9 units. The tuned length is 3.8 median, 2.6 to 6.8 from the
   10th to the 90th chain, and it does not move with the study's tau, 3.8 /
   3.8 / 3.9 below 1, 1 to 5 and above 5, which is NUTS's flatness again.

2. It fixes mu. mu's band at 500 is 0.17 fixed, 0.07 and 0.10 jittered,
   against 0.90 and 0.95 under NUTS at eps 0.1 and 0.2, ends 0.21, lag-1
   +0.29 to +0.31 against NUTS's +0.74. Right for what day 3 implied. But
   the rule aimed at lag-1 zero on mu and landed at +0.29, so the quarter
   period from one unit is short: arccos(0.92) gives a mu period of 15.6
   units against the near twenty day 1 read off the sweep, where mu's lag-1
   crossed zero near five. mu is not a harmonic coordinate and its lag-1 at
   short length falls faster than cos. Short and still enough.

3. Tuned and fixed, it fails on eta_1 instead. The band rejects 90% of
   studies at 500 and eta_1 alone is 0.90, lag-1 -0.43, a shallow W with
   1.15 and 1.21 in the end tenths. The tuned lengths spread over 2.6 to 6.8
   units, and the chains near three units sit on eta's reflection from day 1
   (rho1 -0.95 there) and the ones near six on its return. A per-chain
   length is a fixed length per chain, and day 1's resonances are still
   there for each one. Jitter +/-50% takes eta_1 to 0.17 and 0.05, lag-1
   -0.29 and -0.27, histogram flat to 0.06.

4. The jittered rule passes on the line. Joint band at 500 is 0.40 with the
   widest pick and 0.25 with the slowest, no parameter over 0.17 in either,
   ends 0.20 to 0.21 everywhere. The two rules pick the same coordinate on
   98% of chains, so 0.40 against 0.25 is the noise of 40 studies (about
   0.07 near 0.25) and not a difference between them, the same spread day 2
   saw between four units +/-100% (0.40) and six units +/-50% (0.23). The
   rank mean passes everything, 0.12 to 0.17, as on every day of this
   project.

5. Cost. 45.4 to 45.8 steps a transition, 4498 to 4538 gradients a
   simulation, against 4121 for NUTS and 3960 for fixed four units, and 1000
   more for the warmup at one unit, which the other two did not pay for in
   their totals either (both burn 100 at their own length). So a length
   tuned on mu costs NUTS's price plus 10% and passes where NUTS failed.

The result for a blind rule, and for the project: the U-turn listens to the
many narrow coordinates and stops when they turn; the widest coordinate's
own clock gives a length that serves it, and jitter covers the narrow ones'
resonances, which no single length can avoid. Neither the acceptance rate,
0.996 in every cell, nor the rank mean saw any of it. One right, one wrong
in its number (the quarter period), one not predicted (the fixed per-chain
length failing on eta).

NumPy only. Fixed seeds. About 500 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_period_sweep as d0  # noqa: E402  (puts the eight schools and rank-test modules on the path)
from day1_period_sweep import d1, d2, es, ends, per_param, shape, KEYS, PARAMS, draw_studies, params  # noqa: E402

# Day 1 reseeds on import; reseed again so the tuned studies are fresh draws.
es.RNG = np.random.default_rng(111)
d1.RNG = np.random.default_rng(112)
d2.RNG = np.random.default_rng(113)
RNG = np.random.default_rng(114)

L, SIZES = d1.L, d1.SIZES
EPS, STUDIES, PASS = d0.EPS, d0.STUDIES, d0.PASS
LAG_CHAINS = d0.LAG_CHAINS
PILOT_STEPS = 10                 # warmup phase one at one unit, short enough that every coordinate is still a U
PILOT = 100                      # phase-one transitions; the lag-1 is read on the last READ of them
READ = 80
SETTLE = 50                      # phase-two transitions at the tuned length before the ranks start
T_MIN, T_MAX = 0.5, 25.5         # 25.5 units is day 3's max depth, so the two rules get the same ceiling
NUTS_GRADIENTS = 4121            # day 3, non-centred at eps 0.1, for the cost comparison
RULES = [("widest", 0.0), ("widest", 0.5), ("slowest", 0.5)]
NAMES = ["mu", "log tau"] + [f"eta_{j + 1}" for j in range(8)]


def hmc_counts(z, n):
    """Unit-mass non-centred HMC where chain i runs n[i] leapfrog steps; returns (z, accepted).

    The array is integrated to the longest count and each chain is frozen at its
    own, so no chain's trajectory depends on another's (day 2's scheme).
    """
    pot, grad = es.MODELS["noncentred"]
    p = es.RNG.standard_normal(z.shape)
    h0 = pot(z) + 0.5 * (p * p).sum(1)
    zn = z.copy()
    g = grad(zn)
    for s in range(int(n.max())):
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
    return np.where(acc[:, None], zn, z), acc


def quarter_period(rho, t0):
    """The length at which a coordinate with lag-1 rho at length t0 would have lag-1 zero.

    For a Gaussian coordinate under unit mass the lag-1 autocorrelation at length T
    is cos(omega T), and its IACT as an AR(1) is (1 + rho) / (1 - rho), so reading
    the IACT and reading rho are the same measurement; omega = arccos(rho) / t0 and
    the quarter period pi / (2 omega) is where the chain stops remembering its start.
    """
    rho = np.clip(rho, -0.5, 0.995)
    omega = np.arccos(rho) / t0
    return np.clip(0.5 * np.pi / omega, T_MIN, T_MAX)


def warmup(kind_rule, r):
    """Phase one at one unit, the per-chain choice of coordinate and its quarter period.

    Streams the sums each chain needs for its own variance and lag-1 on the last
    READ transitions, so nothing of size (READ, r, 10) is stored. Returns the
    state, the tuned length in units per chain, the chosen coordinate per chain
    and the per-chain lag-1 of every coordinate.
    """
    z = es.prior_start("noncentred", r)
    n0 = np.full(r, PILOT_STEPS)
    s1 = np.zeros((r, 10))
    s2 = np.zeros((r, 10))
    sx = np.zeros((r, 10))
    prev = None
    for t in range(PILOT):
        z, _ = hmc_counts(z, n0)
        if t >= PILOT - READ:
            s1 += z
            s2 += z * z
            if prev is not None:
                sx += z * prev
            prev = z.copy()
    m = s1 / READ
    var = s2 / READ - m * m
    rho = (sx / (READ - 1) - m * m) / np.maximum(var, 1e-12)
    pick = np.argmax(var if kind_rule == "widest" else rho, axis=1)
    rho_pick = rho[np.arange(r), pick]
    t_star = quarter_period(rho_pick, PILOT_STEPS * EPS)
    return z, t_star, pick, rho


def sbc_tuned(truth, y, rule, width):
    """Warmup, SETTLE transitions at the tuned length, then L with the ranks of the truth accumulated."""
    es.Y = y
    r = y.shape[0]
    z, t_star, pick, rho_w = warmup(rule, r)
    centre = np.maximum(1, np.round(t_star / EPS)).astype(int)
    lo = np.maximum(1, np.round(centre * (1 - width))).astype(int)
    hi = np.round(centre * (1 + width)).astype(int)

    def draw():
        return lo + (RNG.random(r) * (hi - lo + 1)).astype(int)

    for _ in range(SETTLE):
        z, _ = hmc_counts(z, draw())
    below = np.zeros((r, 4), dtype=np.int32)
    acc, steps = 0.0, np.zeros(r)
    kept = np.empty((L, LAG_CHAINS, 3))
    for k in range(L):
        n = draw()
        z, a = hmc_counts(z, n)
        acc += a.mean()
        steps += n
        p = params("noncentred", z)
        kept[k] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return below, acc / L, kept, steps / L, t_star, pick, rho_w


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax
    warm_cost = PILOT * PILOT_STEPS

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    truth, y = draw_studies(r)
    print(f"trajectory length tuned on the chain: non-centred hmc at eps {EPS}, {STUDIES} x {nmax} studies, "
          f"warmup {PILOT} x {PILOT_STEPS} steps (lag-1 on the last {READ}), settle {SETTLE}, L = {L}")

    runs = {}
    print("\n1. the tuned length in units (median, 10th, 90th), the chosen coordinate, acceptance, mean steps, "
          "lag-1 of mu / log tau / eta_1 after tuning")
    for rule, w in RULES:
        below, acc, kept, steps, t_star, pick, rho_w = sbc_tuned(truth, y, rule, w)
        runs[rule, w] = (below.reshape(STUDIES, nmax, 4), steps, t_star)
        rho = d0.lag1(kept)
        share = np.bincount(pick, minlength=10) / r
        top = np.argsort(share)[::-1][:2]
        print(f"  {rule:7s} +/- {100 * w:3.0f}%  T {np.median(t_star):4.1f} ({np.quantile(t_star, 0.1):4.1f}, "
              f"{np.quantile(t_star, 0.9):4.1f})  picks " + ", ".join(f"{NAMES[i]} {share[i]:.2f}" for i in top)
              + f"  acc {acc:.3f}  steps {steps.mean():5.1f}  rho1 {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f}  "
              f"{time.time() - t0:5.0f} s", flush=True)
        if rule == "widest" and w == 0.0:
            print("     warmup lag-1 at one unit, median over chains: "
                  + " ".join(f"{NAMES[i]} {np.median(rho_w[:, i]):+.2f}" for i in range(4)))

    print(f"\n2. share of {STUDIES} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for (rule, w), (rr, _, _) in runs.items():
        cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                          for n in SIZES)
        print(f"  {rule:7s} +/- {100 * w:3.0f}%  {cells}")

    print(f"\n3. each parameter alone at n = {nmax}, band / mean, " + " / ".join(PARAMS)
          + ", and the share of ranks in the end tenths")
    for (rule, w), (rr, _, _) in runs.items():
        b = per_param(rr, stats[nmax], crits[nmax], "band")
        m = per_param(rr, stats[nmax], crits[nmax], "mean")
        flat = rr.reshape(r, 4)
        print(f"  {rule:7s} +/- {100 * w:3.0f}%  " + "  ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m))
              + "   ends " + " ".join(f"{ends(flat[:, p]):.2f}" for p in range(4)))

    print(f"\n4. pooled rank histograms of eta_1 and mu in tenths at n = {nmax}")
    for (rule, w), (rr, _, _) in runs.items():
        print(f"  {rule:7s} +/- {100 * w:3.0f}%  eta_1 {shape(rr, 3)}   mu {shape(rr, 0)}")

    print("\n5. the tuned length by the study's true tau, below 1 / 1-5 / above 5")
    tau = np.exp(truth[:, 1])
    bins = [tau < 1, (tau >= 1) & (tau < 5), tau >= 5]
    for (rule, w), (_, _, t_star) in runs.items():
        print(f"  {rule:7s} +/- {100 * w:3.0f}%  T " + " / ".join(f"{np.median(t_star[b]):4.1f}" for b in bins))

    print(f"\n6. verdict: band on the four together at n = {nmax}, a pass at or under {PASS}; gradients a simulation "
          f"= mean steps x {L}, and with the {warm_cost} of phase one; nuts on day 3 was {NUTS_GRADIENTS}, fixed "
          f"four units {40 * L}")
    for (rule, w), (rr, steps, _) in runs.items():
        share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
        worst = PARAMS[int(np.argmax(per_param(rr, stats[nmax], crits[nmax], "band")))]
        g = steps.mean() * L
        print(f"  {rule:7s} +/- {100 * w:3.0f}%  band {share:.2f}  {'pass' if share <= PASS else 'fail'}  "
              f"worst {worst:8s} {g:7.0f} gradients, {g + warm_cost:7.0f} with warmup")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
