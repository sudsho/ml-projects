"""Day 1 - The ECDF of the ranks under a simultaneous band, against the binned chi-squared.

The SBC project's chi-squared puts 100 rank values into 20 bins and sums the
squared misses, so a histogram that climbs steadily from one end to the other
scores the same as one with the same misses in shuffled order. The ECDF keeps
the order: a tilt adds up as it goes and its excursion is largest in the
middle. Here the ECDF is drawn at the 99 inner rank values, each point scaled
by its binomial sd, and the band is the 95% point of the largest scaled
excursion on 20,000 sets of uniform ranks, so the band and the test are one
object. A plain Kolmogorov-Smirnov distance, the same max with no scaling, is
the third column. Day 3's seven samplers, burn 100 and thin 10, 40 studies,
either of mu and tau rejecting at its own 95% point as day 3 counted it, so
the size to read against is about 10% and the standard error of a row about 5.

1. The band covers what it says. 0.949 to 0.955 on fresh uniform ranks at
   every n from 20 to 500, half-width 4.03 sd at n = 20 and 3.11 to 3.14 from
   n = 100 on. Intervals of 1.96 sd at each point cover all 99 together only
   43% to 51% of the time, so reading an ECDF plot against pointwise bands
   would reject a correct sampler about half the time.

2. The missing log det, the tilt, is where the order pays. The chi-squared
   reaches 65% at 500 simulations, the band 100% at 500 and 45% at 100, 2.2
   to 2.6 times the chi-squared's rate at 100 and 200. It still does
   not get to 80% before 500, so the tilt day 3 could see by eye at 100 is
   visible to the band at 100 less than half the time.

3. The missing Hastings factor was already cheap. 80% at 100 under both, and
   KS gets there at 50 with 90%, where the band has 62%. A shift this large
   is decided in the middle of the ECDF, and scaling the ends up spends the
   band's width where nothing is happening.

4. The dropped half kick is not found by anything here. 7% under chi-squared
   and 20% under the band at 500, both within about two standard errors
   of the size.

5. The sharper test is sharper on the slow sampler too, which is the wrong
   way. The band rejects day 1's slow walk on 78% at 20 simulations and 95%
   at 50, where the chi-squared needs 200 and KS 200. Its ranks are a U,
   piled at both ends, and the scaling that weights the ends is exactly what
   finds a U. The correct walk at step 0.5 is rejected on 12% to 20% from
   n = 50 under the band and 3% to 5% under the chi-squared, a small U from its
   lag. So the band reads autocorrelation as bias more readily than the test
   it replaces, and day 3 of this project has to take the lag out of the
   ranks before any statistic is chosen for its power.

Six predictions written before the run. Two right, one half, three wrong.

- Right: the band covers 95% to within 0.01 at every n. 0.949 to 0.955.
- Half: pointwise 1.96 intervals cover all points under half the time. 43%
  to 46% from n = 50, 51% at n = 20.
- Wrong: the band takes the log det to 80% by 100 simulations. 45%.
- Wrong: the band takes the Hastings bug to 80% by 50. 62%, KS did it.
- Right: nothing catches the half kick, under 30% at 500. 7% and 20%.
- Wrong: every correct sampler stays at 10% or under on the band. The walk
  at 12% to 20%.

NumPy only. Fixed seeds. About 90 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sampler-correctness-sbc"))
import day3_planted_bugs as d3  # noqa: E402

# Same model, samplers, burn and thin as the SBC project's day 3. Its module
# keeps its own generator; reseed it so this file's rows do not depend on
# what day 3 drew before.
d3.RNG = np.random.default_rng(31)
RNG = np.random.default_rng(32)

L, BINS = d3.L, d3.BINS          # 99 draws a simulation, ranks 0..99, 20 bins
GRID = np.arange(1, L + 1) / (L + 1)   # z_i = (i + 1) / (L + 1), i = 0..L-1
SIZES = (20, 50, 100, 200, 500)
REPS = 20000


def rank_counts(ranks):
    """Counts of each rank value 0..L along axis 1, for ranks of shape (S, n)."""
    s = ranks.shape[0]
    idx = ranks + (L + 1) * np.arange(s)[:, None]
    return np.bincount(idx.ravel(), minlength=s * (L + 1)).reshape(s, L + 1)


def chi2_stat(counts):
    """Day 2's Pearson statistic: ranks r go to bin r * BINS // (L + 1), five values a bin."""
    n = counts.sum(1, keepdims=True)
    b = counts.reshape(counts.shape[0], BINS, (L + 1) // BINS).sum(2)
    e = n / BINS
    return ((b - e) ** 2 / e).sum(1)


def ecdf_dev(counts):
    """Standardised ECDF deviations at the L inner grid points, shape (S, L).

    F_hat(z_i) is the share of ranks <= i, and under uniform ranks n F_hat(z_i)
    is Binomial(n, z_i), so dividing by sqrt(z (1 - z) / n) puts every point
    on the same scale and the ends of the ECDF count as much as the middle.
    """
    n = counts.sum(1, keepdims=True)
    f = np.cumsum(counts, 1)[:, :L] / n
    return (f - GRID) / np.sqrt(GRID * (1 - GRID) / n)


def band_stat(counts):
    """The widest standardised excursion of the ECDF, the statistic the band is drawn from."""
    return np.abs(ecdf_dev(counts)).max(1)


def ks_stat(counts):
    """Plain Kolmogorov-Smirnov distance, the same max but unweighted, for comparison."""
    n = counts.sum(1, keepdims=True)
    return np.abs(np.cumsum(counts, 1)[:, :L] / n - GRID).max(1)


STATS = {"chi2": chi2_stat, "band": band_stat, "ks": ks_stat}


def null_counts(n, reps=REPS):
    """Rank counts of reps studies of n simulations from a correct sampler, ranks uniform on 0..L.

    Every critical value and coverage row in the file is read off these, and
    day 2 calibrates its mean and variance tests with it too.
    """
    return rank_counts(RNG.integers(0, L + 1, (reps, n)))


def critical_values(n):
    """95% points of each statistic on REPS sets of n uniform ranks, by simulation.

    The band at level 0.95 is GRID +- c sqrt(z (1 - z) / n) with c the band
    statistic's 95% point, so the band and the test are the same object.
    """
    c = null_counts(n)
    return {k: np.quantile(f(c), 0.95) for k, f in STATS.items()}


def coverage(n, crit):
    """Simultaneous coverage of the band and of 1.96 pointwise intervals, on fresh null sets."""
    dev = ecdf_dev(null_counts(n))
    band = (np.abs(dev).max(1) <= crit["band"]).mean()
    pointwise = (np.abs(dev).max(1) <= 1.96).mean()
    return band, pointwise


def reject_any(ranks, crit):
    """Share of studies where either parameter's statistic passes its 95% point."""
    out = {}
    for k, f in STATS.items():
        hit = np.zeros(ranks.shape[0], dtype=bool)
        for p in range(2):
            hit |= f(rank_counts(ranks[:, :, p])) > crit[k]
        out[k] = hit.mean()
    return out


def first_size(row, key, level=0.8):
    """Smallest simulation count in SIZES at which the rejection share reaches level."""
    for n in SIZES:
        if row[n][key] >= level:
            return str(n)
    return f">{SIZES[-1]}"


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    studies = 40

    print("1. the band's coverage on uniform ranks, fresh sets, against 1.96 pointwise")
    crits = {}
    for n in SIZES:
        crits[n] = critical_values(n)
        band, pw = coverage(n, crits[n])
        print(f"  n={n:3d}  band half-width {crits[n]['band']:.2f} sd  "
              f"band covers {band:.3f}   pointwise covers all {L} points {pw:.3f}")

    print(f"\n2. share of {studies} studies rejected, chi2 / band / ks, mu or tau")
    table = {}
    for kind in d3.SAMPLERS:
        rk = d3.sbc_ranks(kind, studies, SIZES[-1])
        table[kind] = {n: reject_any(rk[:, :n], crits[n]) for n in SIZES}
        cells = "  ".join(
            f"n={n}: {table[kind][n]['chi2']:.2f}/{table[kind][n]['band']:.2f}/{table[kind][n]['ks']:.2f}"
            for n in SIZES)
        print(f"  {kind:8s} {cells}")

    print("\n3. simulations to reach 80% rejections, chi2 / band / ks")
    for kind in d3.SAMPLERS:
        row = table[kind]
        print(f"  {kind:8s} {first_size(row, 'chi2'):>5s} {first_size(row, 'band'):>5s} "
              f"{first_size(row, 'ks'):>5s}   transitions/sim {d3.BURN + L * d3.THIN:,}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
