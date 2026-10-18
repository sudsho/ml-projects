"""Day 3 - The correct sampler on fresh studies, eta ranks against theta ranks, and what a divergence costs a chain.

Day 2 left three things open. The non-centred sampler's mean z on log tau
sat at -0.4 at every thin on one set of studies, a 1-in-80 draw or something
small in the ranking of the scale. The textbook fix failed the band at thin
1 on mu, and whether a longer trajectory does what thinning did was not
tried. And the divergence count fired on four chains in five, so whether it
sorts anything had to wait for thin 10, where some centred studies pass at
20 simulations. This file draws 40 fresh studies of 500, runs both samplers
at eps 0.1 as before, and ranks eta_1 = (theta_1 - mu) / tau beside the
three parameters of day 2, the non-centred sampler's own coordinate against
the derived one, and under centred the other way round. Then the non-centred
sampler at eps 0.2 and 0.4 at thin 1 only, 20 leapfrog steps still, so the
trajectory is two and four times longer and the chain is not thinned.
Finally, centred at thin 10: the divergence count of each study against its
band rejection at 20 and 50 simulations, and the ranks of chains with a
divergence against the ranks of chains without.

1. The -0.4 was a draw. On fresh studies log tau's mean z under the
   non-centred sampler is -0.00, +0.02 and -0.02 at the three thins and the
   pooled histogram sits between 0.96 and 1.06. Nothing is small in the
   ranking of the scale.

2. The correct sampler at thin 1 fails on mu and on nothing else, and the
   sampled coordinate is the one that passes. The band rejects 97% of
   studies on mu at 500, ends 1.24 and 1.27, against 12% on log tau, 15% on
   theta_1 and 0% on eta_1, whose histogram is flat, 0.96 to 1.05. theta_1 =
   mu + tau eta_1 carries mu's U into its own, 1.06 and 1.07. At eps 0.1 the
   20-step trajectory is two units long and mu's posterior is wider than
   that, so unit-mass HMC walks mu a step at a time and the band reads the
   lag; IACT(log tau) is 2.1 and says nothing about it.

3. A longer trajectory does what thinning did, and the window is narrow.
   At eps 0.2, four units, thin 1 passes everything at the nominal rate of
   a four-parameter test, band 12% to 23% and mu alone 5% at 100 and 10% at
   500, lag-1 +0.06, acceptance 0.98, no divergences, so one transition at
   eps 0.2 buys what five at eps 0.1 did. At eps 0.4 acceptance falls to
   0.72, the non-centred sampler diverges for the first time in this
   project, 0.11% of proposals on 3.3% of chains, lag-1 is +0.57 from
   repeated states, and the band rejects 97% of studies on log tau at 500,
   ends 1.15 and 1.12: a U from rejections where thin 1's was a U from a
   lag, and the same shape from the opposite cause.

4. Under centred HMC the derived coordinate hides the failure the sampled
   ones show. At thin 1 mu's ends are 2.30 and 2.33, log tau's 2.13 and
   2.26, theta_1's 2.13 and 2.18, and eta_1's 1.40 and 1.41; the band on
   eta_1 alone rejects 55% at 100 against 100% on each of the other three,
   and at thin 10 15% against 38% to 60%. A stuck chain holds theta_1 and
   mu together and (theta_1 - mu) / tau cancels most of what is stuck. So
   on the correct sampler the derived parameter carries a lag its sampled
   parent does not have, and on the broken one it drops a failure its
   parents have; ranking what the sampler carries is not safe either way,
   rank both.

5. The divergence count sorts the studies on the scale, weakly, and only
   as the band gains power. At thin 10 and 20 simulations the band rejects
   30% of studies on log tau, and those have a divergence rate of 3.35% of
   proposals against 1.63% among the studies it passes, double, with a rank
   correlation between rate and band statistic of +0.13; at 50 it is 2.86%
   against 1.70% and +0.28, at 100 2.76% against 1.95% and +0.34. On mu
   there is nothing, 1.88% against 2.27% at 50 and a correlation of 0.00.
   A flag that doubles between the rejected and the passed and ranks the
   studies at 0.13 is not a stand-in for the test.

6. A divergence costs the chain nothing on the scale. Chains with one put
   log tau's rank in the end tenths 26% of the time at thin 10 against 27%
   for chains without, 44% against 45% at thin 1, and on mu and theta_1
   four to five points more, 0.26 against 0.22 and 0.25 against 0.20. The
   10% of chains with 17 or more divergences are the stuck ones, theta_1 at
   0.41 against 0.20, mu 0.38, eta_1 0.35 and log tau 0.33, and their truth
   has a slightly smaller tau, median 3.23 against 4.15, neck 16.8% against
   13.6%. So what marks a stuck chain is the count and not the flag, and it
   marks it on the school effect first and the scale last, the opposite of
   the quantity the divergence is said to diagnose.

Six predictions written before the run. Two right, one half, three wrong.

- Right: log tau's mean z on the fresh non-centred studies is within 0.2 of
  zero at every thin. -0.00, +0.02, -0.02.
- Half: under non-centred at thin 1, eta_1 and theta_1 both pass the band
  under 15% at every n while mu fails above 70% at 500. eta_1 0% to 3%, mu
  97%, theta_1 exactly 15% at 100 and 500.
- Wrong: under centred at thin 1, the derived eta_1 is a U as deep as
  theta_1's, both end tenths above 1.9. 1.40 and 1.41 against 2.13 and 2.18.
- Right: centred at thin 10, n = 20, the studies the band rejects on log tau
  have a divergence rate at least 20% higher than the studies it passes.
  3.35% against 1.63%.
- Wrong: per chain at thin 10, centred chains with a divergence put log
  tau's rank in the end tenths at least 10 points more often than chains
  without. 0.26 against 0.27.
- Wrong: non-centred at eps 0.4, thin 1, passes the band on mu under 15% at
  every n with no divergences. mu 45% at 500 and log tau 97%, 0.11% of
  proposals diverge, acceptance 0.72.

NumPy only. Fixed seeds. About 300 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rank-tests-sharper-sbc"))
sys.path.insert(0, os.path.join(HERE, "..", "sampler-correctness-sbc"))
import day1_centred_vs_noncentred as es  # noqa: E402
import day1_ecdf_bands as d1  # noqa: E402
import day2_shape_tests as d2  # noqa: E402
from day2_sbc_with_data import iact_summary, per_param, shape  # noqa: E402

# Day 2's module reseeds the imported generators when it loads; reseed again
# so these studies are fresh and share nothing with the 16th's.
es.RNG = np.random.default_rng(61)
d1.RNG = np.random.default_rng(62)
d2.RNG = np.random.default_rng(63)
RNG = np.random.default_rng(64)

L, SIZES = d1.L, d1.SIZES
EPS = 0.1
BURN = 100
PILOT = 990
THINS = (1, 5, 10)
KEYS = ("chi2", "band", "mean", "var")
PARAMS = ("mu", "log tau", "theta_1", "eta_1")
KINDS = ("noncentred", "centred")
LONG_EPS = (0.2, 0.4)            # non-centred at thin 1 only, the trajectory lengthened instead of the chain thinned
IACT_CHAINS = 200
TENTHS = 10


def draw_studies(r):
    """r prior draws of the hierarchy and a data set from each. Returns the truth (r, 4) and y (r, J)."""
    mu = es.MU_SD * es.RNG.standard_normal(r)
    tau = es.TAU_SD * np.abs(es.RNG.standard_normal(r))
    eta = es.RNG.standard_normal((r, es.J))
    theta = mu[:, None] + tau[:, None] * eta
    y = theta + es.SIGMA * es.RNG.standard_normal((r, es.J))
    return np.column_stack([mu, np.log(tau), theta[:, 0], eta[:, 0]]), y


def params(kind, z):
    """mu, log tau, theta_1 and eta_1 from either state; one of the last two is sampled, the other derived."""
    mu, u = z[:, 0], z[:, 1]
    if kind == "centred":
        th1 = z[:, 2]
        eta1 = (th1 - mu) / np.exp(u)
    else:
        eta1 = z[:, 2]
        th1 = mu + np.exp(u) * eta1
    return np.column_stack([mu, u, th1, eta1])


def sbc_pilot(kind, truth, y, eps, thins, pilot):
    """Burn, then pilot transitions with the ranks of the truth accumulated at each thin.

    Returns ranks per thin (r, 4), the divergence count per chain, the mean
    acceptance, and the log tau pilot of IACT_CHAINS chains at thin 1.
    """
    es.Y = y                     # day 1's potentials read the data from the module; (r, J) broadcasts per chain
    r = y.shape[0]
    z = es.prior_start(kind, r)
    for _ in range(BURN):
        z, _, _ = es.hmc(kind, z, eps)
    below = {thin: np.zeros((r, 4), dtype=np.int32) for thin in thins}
    div = np.zeros(r)
    acc = 0.0
    kept = np.empty((pilot, IACT_CHAINS))
    for k in range(1, pilot + 1):
        z, a, d = es.hmc(kind, z, eps)
        div += d
        acc += a.mean()
        kept[k - 1] = z[:IACT_CHAINS, 1]
        p = params(kind, z)
        for thin in thins:
            if k % thin == 0 and k // thin <= L:
                below[thin] += p < truth
    return below, div, acc / pilot, kept


def ends(ranks):
    """Share of ranks in the two end tenths, the U's own number; 0.2 under uniform ranks."""
    g = ranks * TENTHS // (L + 1)
    return np.mean((g == 0) | (g == TENTHS - 1))


def spearman(a, b):
    """Rank correlation of two study-level arrays."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def print_info(label, acc, div, kept):
    med, q90, bad, rho1 = iact_summary(kept)
    print(f"  {label:22s} acc {acc:.3f} div {100 * div.mean() / kept.shape[0]:5.2f}% "
          f"of proposals, {100 * (div > 0).mean():5.1f}% of chains  iact median {med:5.1f} 90% {q90:6.1f}  "
          f"nan guard {bad:.4f}  rho1 {rho1:+.2f}")


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    studies = 40
    nmax = SIZES[-1]
    r = studies * nmax

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}
    sd_m, _ = d2.null_sds(nmax)

    truth, y = draw_studies(r)
    print(f"eight schools, fresh studies: {studies} x {nmax}, {es.STEPS} leapfrog steps, burn {BURN}, L = {L}")
    print(f"  the prior studies: tau < {es.NECK:.0f} on {100 * np.mean(np.exp(truth[:, 1]) < es.NECK):.1f}%, "
          f"median tau {np.median(np.exp(truth[:, 1])):.2f}, |eta_1| median {np.median(np.abs(truth[:, 3])):.2f}")

    runs = {}                    # label -> (ranks per thin (studies, nmax, 4), div per chain)
    print("\n1. the pilots: acceptance, divergences per proposal and chains with one, IACT of log tau, nan guard, lag-1")
    for kind in KINDS:
        below, div, acc, kept = sbc_pilot(kind, truth, y, EPS, THINS, PILOT)
        label = f"{kind} eps {EPS}"
        runs[label] = ({thin: below[thin].reshape(studies, nmax, 4) for thin in THINS}, div)
        print_info(label, acc, div, kept)
    for eps in LONG_EPS:
        below, div, acc, kept = sbc_pilot("noncentred", truth, y, eps, (1,), L)
        label = f"noncentred eps {eps}"
        runs[label] = ({1: below[1].reshape(studies, nmax, 4)}, div)
        print_info(label, acc, div, kept)
    print(f"  {time.time() - t0:.0f} s", flush=True)

    print(f"\n2. share of {studies} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for label, (ranks, _) in runs.items():
        for thin, rr in ranks.items():
            cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                              for n in SIZES)
            print(f"  {label:20s} thin {thin:2d} {cells}")

    print(f"\n3. each parameter alone at n = 100 and {nmax}, band / mean, " + " / ".join(PARAMS))
    for label, (ranks, _) in runs.items():
        for thin, rr in ranks.items():
            cells = []
            for n in (100, nmax):
                b = per_param(rr[:, :n], stats[n], crits[n], "band")
                m = per_param(rr[:, :n], stats[n], crits[n], "mean")
                cells.append(f"n={n}: " + " ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m)))
            print(f"  {label:20s} thin {thin:2d} " + "  ".join(cells))

    print(f"\n4. pooled rank histograms in tenths at n = {nmax}; log tau's mean z averaged over studies, day 2 had -0.40")
    for label, (ranks, _) in runs.items():
        for thin, rr in ranks.items():
            mz = ((d2.rank_mean(d1.rank_counts(rr[:, :, 1])) - L / 2) / sd_m).mean()
            print(f"  {label:20s} thin {thin:2d} mu      {shape(rr, 0)}")
            print(f"  {'':20s} {'':7s} log tau {shape(rr, 1)}   mean z {mz:+.2f}")
            print(f"  {'':20s} {'':7s} theta_1 {shape(rr, 2)}")
            print(f"  {'':20s} {'':7s} eta_1   {shape(rr, 3)}")

    print("\n5. centred at thin 10: the divergence count of a study against its band rejection on log tau and on mu")
    ranks, div = runs[f"centred eps {EPS}"]
    div_s = div.reshape(studies, nmax)
    for n in (20, 50, 100):
        rr = ranks[10][:, :n]
        rate = div_s[:, :n].mean(1) / PILOT * 100
        for p, name in ((1, "log tau"), (0, "mu")):
            b = stats[n]["band"](d1.rank_counts(rr[:, :, p]))
            rej = b > crits[n]["band"]
            if 0 < rej.sum() < studies:
                split = f"div {rate[rej].mean():.2f}% rejected vs {rate[~rej].mean():.2f}% passed"
            else:
                split = f"div {rate.mean():.2f}%, no split"
            print(f"  n={n:3d} {name:7s} band rejects {rej.mean():.2f}  {split}  "
                  f"spearman(div rate, band stat) {spearman(rate, b):+.2f}")

    print("\n6. centred, chains with a divergence against chains without: share of ranks in the end tenths, 0.20 if uniform")
    has = div > 0
    for thin in THINS:
        rr = ranks[thin].reshape(r, 4)
        cells = "  ".join(f"{name} {ends(rr[has, p]):.2f} vs {ends(rr[~has, p]):.2f}" for p, name in enumerate(PARAMS))
        print(f"  thin {thin:2d} {cells}")
    many = div >= np.quantile(div, 0.9)
    rr = ranks[10].reshape(r, 4)
    print(f"  thin 10, the 10% of chains with the most divergences ({div[many].min():.0f}+) vs none: "
          + "  ".join(f"{name} {ends(rr[many, p]):.2f} vs {ends(rr[~has, p]):.2f}" for p, name in enumerate(PARAMS)))
    nd_r = np.exp(truth[~has, 1])
    d_r = np.exp(truth[has, 1])
    print(f"  the truth's tau: median {np.median(d_r):.2f} on chains with a divergence, {np.median(nd_r):.2f} without; "
          f"tau < {es.NECK:.0f} on {100 * np.mean(d_r < es.NECK):.1f}% and {100 * np.mean(nd_r < es.NECK):.1f}%")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
