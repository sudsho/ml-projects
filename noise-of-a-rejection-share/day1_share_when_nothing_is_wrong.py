"""Day 1 - The share when nothing is wrong.

Three projects read a sampler as passing when the band rejected 0.25 or
fewer of 40 studies on any of three parameters, and failing near 0.40, with
a noise put at 0.07 that nobody measured. Before any sampler, this file
measures the share when nothing can be wrong: ranks drawn uniform by
construction, 2000 sets of 40 studies, each study up to 500 simulations,
the band's 95% points read off the rank tests project's uniform sets as
before. The three parameters' truths are tied through one Gaussian copula at
correlation 0, 0.5 and 0.9, since mu and theta_1 of one simulation move
together. The 0.07 is what the binomial gives at a share of 0.25, so it
was a guess at the sampler's noise, not the null's.

1. The share is binomial and nothing more. Each parameter alone rejects
   0.048 to 0.053 at every n, sd 0.034 to 0.035 against the binomial's
   0.034 to 0.035. The joint at rho 0 is 0.139 to 0.148 against the nominal
   0.143, sd 0.054 to 0.056 against 0.055. The 40 studies of a set are
   independent the way that sd assumes, and the 95% points carry no
   noise worth counting.

2. Tied parameters make the joint smaller, not noisier. At rho 0.5 the
   joint is 0.132 to 0.141 and at 0.9 it is 0.111 to 0.119, sd 0.050 to
   0.051, still binomial at its own mean. The parameters' shares in one set
   correlate +0.05 and +0.23. So the old line of 0.25 is further from a
   correct sampler on eight schools than its 0.19 comment said.

3. A correct sampler almost never reads on the line. Above 0.25 on 2.0% of
   sets at 500 under rho 0, 1.6% at 0.5, 0.5% at 0.9, the middle 90%
   0.05 to 0.25 and the worst of 2000 sets 0.325 to 0.425. A reading of
   0.40 is not a correct sampler on a bad draw.

4. Sizes in one set are not independent readings. The studies at n are the
   first n simulations of the 500, so a set's shares at neighbouring sizes
   correlate +0.18 to +0.31, and 20 against 500 is +0.01 to +0.04. A row
   that fails at 200 and 500 is less than two looks.

What this does not settle is the 0.2 the last project saw. A binomial at
0.5 has sd 0.079 at 40 studies, so two sets of one sampler differ by 0.11
in sd and 0.60 against 0.42 is 1.6 of those; 0.25 against 0.47 is 2.0. Both
are what independent studies would give at a share in the middle, where the
binomial is widest. Studies whose own chances differ would make the spread
narrower, so if day 2 finds it wider than the binomial, something is shared
between the studies of a set.

NumPy only. Fixed seeds. About 90 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rank-tests-sharper-sbc"))
import day1_ecdf_bands as d1  # noqa: E402  (rank counts, the band statistic, uniform null counts)

# day1_ecdf_bands reseeds the SBC project's generator on import; reseed its own
# so the critical values here are not the ones that project drew.
d1.RNG = np.random.default_rng(171)
RNG = np.random.default_rng(172)

L, SIZES = d1.L, d1.SIZES        # 99 draws a simulation, ranks 0..99; 20 to 500 simulations
STUDIES = 40                     # what every verdict in three projects was read on
PARAMS = 3                       # mu, log tau and one of theta_1 / eta_1, as in the eight schools rows
PASS = 0.25                      # the line those projects drew
SETS = 2000                      # sets of 40 studies, each from fresh simulations
CHUNK = 100                      # sets per batch, so the ranks fit in memory
RHOS = (0.0, 0.5, 0.9)           # correlation of the truths' quantiles across the three parameters


def correct_ranks(studies, nsim, rho):
    """Ranks a correct sampler gives, shape (studies, nsim, PARAMS), uniform on 0..L for every parameter.

    Each parameter's truth sits at quantile u of its own posterior and the
    rank is the count of L posterior draws below it, Binomial(L, u), which is
    uniform on 0..L once u is. The three u's come from one Gaussian copula with
    pairwise correlation rho, so the parameters of one simulation can move
    together the way mu and theta_1 do; given u the draws are independent,
    which leaves out the draws' own correlation.
    """
    z = RNG.standard_normal((studies, nsim, PARAMS))
    if rho > 0:
        z = np.sqrt(1 - rho) * z + np.sqrt(rho) * RNG.standard_normal((studies, nsim, 1))
    return RNG.binomial(L, _phi(z))


def _phi(z):
    """Standard normal cdf, vectorised through the error function's rational approximation in NumPy."""
    # Abramowitz and Stegun 7.1.26 is good to 1.5e-7, far below one rank's 0.01.
    x = np.abs(z) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1 - poly * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * erf)


def crits():
    """The band's 95% point at every n, from d1's 20,000 uniform sets, as every earlier project read it."""
    return {n: np.quantile(d1.band_stat(d1.null_counts(n)), 0.95) for n in SIZES}


def shares(rho, crit, sets=SETS):
    """Rejection shares over sets of STUDIES, at every n, joint (any parameter) and per parameter.

    The studies at n are the first n simulations of the 500, the prefix the
    earlier projects took, so a set's shares at two sizes share simulations.
    Returns joint of shape (sets, len(SIZES)) and alone of shape (sets, len(SIZES), PARAMS).
    """
    joint = np.empty((sets, len(SIZES)))
    alone = np.empty((sets, len(SIZES), PARAMS))
    nmax = SIZES[-1]
    for s0 in range(0, sets, CHUNK):
        k = min(CHUNK, sets - s0)
        rr = correct_ranks(k * STUDIES, nmax, rho)
        for j, n in enumerate(SIZES):
            hit = np.stack([d1.band_stat(d1.rank_counts(rr[:, :n, p])) > crit[n] for p in range(PARAMS)], -1)
            hit = hit.reshape(k, STUDIES, PARAMS)
            alone[s0:s0 + k, j] = hit.mean(1)
            joint[s0:s0 + k, j] = hit.any(2).mean(1)
    return joint, alone


def binom_sd(p, n=STUDIES):
    """sd of a share of n independent studies each rejected with probability p."""
    return np.sqrt(p * (1 - p) / n)


def main():
    t0 = time.time()
    crit = crits()
    nominal = 1 - 0.95 ** PARAMS
    print(f"uniform ranks by construction, {SETS} sets of {STUDIES} studies, L = {L}, {PARAMS} parameters")
    print("  band 95% points: " + "  ".join(f"n={n} {crit[n]:.2f}" for n in SIZES))
    print(f"  joint nominal 1 - 0.95^{PARAMS} = {nominal:.3f}, binomial sd at 40 {binom_sd(nominal):.3f}")

    out = {}
    for rho in RHOS:
        joint, alone = shares(rho, crit)
        out[rho] = (joint, alone)
        print(f"\nrho {rho:.1f}")
        print("     n   alone mean  alone sd (binom)  joint mean  joint sd (binom at its mean)  "
              "q05   q95  P(>0.25)  max")
        for j, n in enumerate(SIZES):
            a = alone[:, j].ravel()
            g = joint[:, j]
            print(f"  {n:4d}   {a.mean():.3f}      {a.std():.3f} ({binom_sd(a.mean()):.3f})    "
                  f"{g.mean():.3f}       {g.std():.3f} ({binom_sd(g.mean()):.3f})               "
                  f"{np.quantile(g, 0.05):.3f} {np.quantile(g, 0.95):.3f}  {(g > PASS).mean():.3f}  {g.max():.3f}")
        # prefixes: the share at 200 and the share at 500 of one set are read off the same simulations
        c = np.corrcoef(joint.T)
        print("  corr of a set's joint share across n (prefixes): "
              + "  ".join(f"{SIZES[i]}-{SIZES[i + 1]} {c[i, i + 1]:+.2f}" for i in range(len(SIZES) - 1))
              + f"  20-500 {c[0, -1]:+.2f}")
        # parameters: within one set, do the three per-parameter shares move together?
        a5 = alone[:, -1]
        cp = np.corrcoef(a5.T)
        print(f"  corr between parameters' shares at 500: {cp[0, 1]:+.2f} {cp[0, 2]:+.2f} {cp[1, 2]:+.2f}")

    # how wide one reading is at 40 studies, from the independent case
    g = out[0.0][0][:, -1]
    print(f"\nat 500, rho 0: a correct sampler reads above {PASS} on {100 * (g > PASS).mean():.1f}% of sets,"
          f" the middle 90% is {np.quantile(g, 0.05):.3f} to {np.quantile(g, 0.95):.3f}")
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
