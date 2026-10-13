"""Day 2 - Tests aimed at a shape: the rank mean for a shift, the rank variance for a width.

Day 1's band and the chi-squared are omnibus tests, they spend their size on
every way a histogram can be wrong. SBC day 3 already said which way each bug
goes: the missing log det and the missing Hastings factor tilt tau's ranks,
the dropped half kick humps them, the slow walk piles them at both ends. A
tilt moves the rank mean and a hump or a U moves the rank variance, so each
gets a statistic of one number, |mean - L / 2| and |var - V| with V the
discrete uniform's (L + 1)^2 - 1 over 12, and a third that takes the larger
of the two once each is divided by its null sd. All five are calibrated the
way day 1 calibrated the band, the 95% point of 20,000 sets of uniform ranks.
Day 3's seven samplers, burn 100 and thin 10, 40 studies, either parameter.

A last row is a bug that is neither, made at the level of the ranks: the
truth is N(0, 1) and the 99 draws come from a gamma of shape 4 moved and
scaled to mean 0 and variance 1, so the first two moments are right and the
skew is 1 where it should be 0.

1. All five hold 3.9% to 5.2% on one parameter of uniform ranks at every n.
   On the correct samplers, either parameter, they sit at 3% to 20%, the
   chi-squared highest at 20% on HMC at 100, so a row is read against about
   10% with a standard error near 5, as on day 1.

2. The rank mean is the cheapest test of a shift. The Hastings bug reaches
   97% at 50 simulations under it, against 72% for the band and 47% for the
   chi-squared, and the log det 55% at 100 and 78% at 200, against 33% and
   62% for the band. Both still need 500 for the log det to pass 80%.

3. The rank mean does not see the slow walk at all. 17%, 10%, 10% and 12%
   from 50 to 500 simulations, inside the size, where the band, the variance
   and the chi-squared all pass 80% by 100. A chain that never crosses the
   truth in a run puts the ranks at both ends about equally, which moves the
   variance by 9.4 null sds and the mean by 0.4. So the one statistic that
   reads lag as bias least is also the cheapest on both shift bugs.

4. The rank variance is the right test for the half kick and still not
   enough. The hump is there, tau's variance z averages -1.48 at 500, and the
   variance test rejects 38% at 500, the most anything has reached on it,
   against 25% for the band and 15% for the chi-squared. On a tilt it gives
   up nearly everything: the log det at 23% at 500.

5. The bug that is neither is found by the omnibus tests and not by the
   aimed ones. The band takes the skewed draws to 85% at 100 and the
   chi-squared to 97% at 200, the mean stays at 25% to 28% and the variance
   at 7% to 28% until 500, and they reach 57% and 55% there. The skew does move both, mean z
   +1.49 and variance z +1.20 on tau, but by too little a study to decide.

Six predictions written before the run. One right, one half, four wrong.

- Wrong: the mean and variance tests hold 10% or under on the correct
  samplers at every n. 15% for the mean on HMC at 500 and on the Gibbs walk at
  50, 15% to 17% for the variance on RMHMC, inside two standard errors, as
  the chi-squared's 20% is.
- Half: the mean test takes the log det to 80% by 200. 78% at 200.
- Right: the mean test takes the Hastings bug to 80% by 50. 97%.
- Wrong: the variance test catches the half kick on 80% by 500. 38%.
- Wrong: the variance test rejects the slow walk at least as early as the
  band, 80% at 20. 70% at 50 and 93% at 100, where the band had 97% at 50.
- Wrong: the skewed draws pass the mean and the variance, under 30% at 500,
  and only the band catches them. The band does, 85% at 100, but the mean and
  the variance reach 57% and 55% at 500.

Day 3 reads this the other way round from the plan. Before taking the lag
out of the ranks, the rank mean already passes the slow walk and catches both
shift bugs sooner than the band does. What it cannot see is a change of width,
and the half kick is the only planted bug that is one.

NumPy only. Fixed seeds. About 90 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sampler-correctness-sbc"))
import day1_ecdf_bands as d1  # noqa: E402
import day3_planted_bugs as d3  # noqa: E402

# Importing day 1 reseeds day 3's generator; reseed both again here so this
# file's rows do not share draws with day 1's.
d3.RNG = np.random.default_rng(41)
d1.RNG = np.random.default_rng(42)
RNG = np.random.default_rng(43)

L, SIZES, REPS = d1.L, d1.SIZES, d1.REPS
VALUES = np.arange(L + 1)
V_UNIF = ((L + 1) ** 2 - 1) / 12.0


def rank_mean(counts):
    """Mean rank of each study, from counts of shape (S, L + 1)."""
    return (counts * VALUES).sum(1) / counts.sum(1)


def rank_var(counts):
    """Population variance of the ranks of each study, about their own mean."""
    n = counts.sum(1)
    m = rank_mean(counts)
    return (counts * VALUES ** 2).sum(1) / n - m ** 2


def null_sds(n):
    """Null sds of the mean and variance statistics at n, by simulation, for the combined test."""
    c = d1.null_counts(n)
    return rank_mean(c).std(), rank_var(c).std()


def shape_stats(n):
    """The three shape statistics at n, each a function of counts; the max one needs null sds."""
    sd_m, sd_v = null_sds(n)
    mean_z = lambda c: (rank_mean(c) - L / 2) / sd_m           # noqa: E731
    var_z = lambda c: (rank_var(c) - V_UNIF) / sd_v            # noqa: E731
    return {
        "mean": lambda c: np.abs(mean_z(c)),
        "var": lambda c: np.abs(var_z(c)),
        "max": lambda c: np.maximum(np.abs(mean_z(c)), np.abs(var_z(c))),
        "band": d1.band_stat,
        "chi2": d1.chi2_stat,
    }, var_z


def critical_values(stats):
    """95% point of each statistic in stats on fresh uniform ranks at stats["n"].

    stats carries its own n so the max statistic's null sds and its critical
    value come from the same simulation count. Day 3 reads its chi-squared,
    band and mean thresholds off this too.
    """
    c = d1.null_counts(stats["n"])
    return {k: np.quantile(f(c), 0.95) for k, f in stats.items() if k != "n"}


def reject_any(ranks, stats, crit):
    """Share of studies where either parameter's statistic passes its 95% point."""
    out = {}
    for k in crit:
        hit = np.zeros(ranks.shape[0], dtype=bool)
        for p in range(ranks.shape[2]):
            hit |= stats[k](d1.rank_counts(ranks[:, :, p])) > crit[k]
        out[k] = hit.mean()
    return out


def skewed_ranks(studies, nsim, shape=4.0):
    """Ranks of an N(0, 1) truth among L gamma draws standardised to mean 0, variance 1.

    The draws have the truth's first two moments and skew 2 / sqrt(shape), so a
    test of the mean or the variance alone is looking at the wrong thing by
    construction. Two independent parameters, to match the samplers' rows.
    """
    t = RNG.standard_normal((studies, nsim, 2))
    g = (RNG.gamma(shape, 1.0, (studies, nsim, 2, L)) - shape) / np.sqrt(shape)
    return (g < t[..., None]).sum(-1)


def first_size(row, key, level=0.8):
    for n in SIZES:
        if row[n][key] >= level:
            return str(n)
    return f">{SIZES[-1]}"


KEYS = ("chi2", "band", "mean", "var", "max")


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    studies = 40

    stats, var_z, crits = {}, {}, {}
    for n in SIZES:
        s, var_z[n] = shape_stats(n)
        stats[n] = s
        crits[n] = critical_values(dict(s, n=n))

    print("1. size of each test on fresh uniform ranks, one parameter, 20,000 sets")
    for n in SIZES:
        c = d1.null_counts(n)
        cells = "  ".join(f"{k} {(stats[n][k](c) > crits[n][k]).mean():.3f}" for k in KEYS)
        print(f"  n={n:3d}  {cells}")

    ranks = {kind: d3.sbc_ranks(kind, studies, SIZES[-1]) for kind in d3.SAMPLERS}
    ranks["skew"] = skewed_ranks(studies, SIZES[-1])

    print(f"\n2. share of {studies} studies rejected, either parameter, " + " / ".join(KEYS))
    table = {}
    for kind, rk in ranks.items():
        table[kind] = {n: reject_any(rk[:, :n], stats[n], crits[n]) for n in SIZES}
        cells = "  ".join(f"n={n}: " + "/".join(f"{table[kind][n][k]:.2f}" for k in KEYS)
                          for n in (50, 100, 200, 500))
        print(f"  {kind:8s} {cells}")

    print("\n3. simulations to reach 80% rejections, " + " / ".join(KEYS))
    for kind in ranks:
        print(f"  {kind:8s} " + " ".join(f"{first_size(table[kind], k):>5s}" for k in KEYS))

    print("\n4. which way tau's ranks move at n = 500: mean z and variance z, averaged over studies")
    n = SIZES[-1]
    for kind, rk in ranks.items():
        c = d1.rank_counts(rk[:, :n, 1])
        sd_m = (rank_mean(d1.null_counts(n, 4000))).std()
        mz = ((rank_mean(c) - L / 2) / sd_m).mean()
        print(f"  {kind:8s} mean z {mz:+6.2f}   var z {var_z[n](c).mean():+6.2f}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
