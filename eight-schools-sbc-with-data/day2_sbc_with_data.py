"""Day 2 - SBC with data in it: ranks of mu, log tau and theta_1 under both parameterisations.

Every rank test so far has been against the prior. Here each simulation
draws the hierarchy from its prior, mu ~ N(0, 25), tau half-normal(5),
theta_j = mu + tau eta_j, then eight observations y_j ~ N(theta_j, sigma_j^2)
with the real study's sigmas, and the chain, started from an independent
prior draw, has to find that simulation's posterior. Both of day 1's samplers
at eps 0.1, 20 leapfrog steps, burn 100 and a 990-transition pilot read at
thin 1, 5 and 10, 40 studies of 500 simulations each. The truth's rank among
99 draws of mu, of log tau and of theta_1 goes to the chi-squared, the band,
the rank mean and the rank variance with the sharper rank tests project's
critical values, rejected when any of the three parameters is.

1. Centred HMC with data in the model is the slow walk, and it is a U, not
   a tilt. The band rejects every study at 20 simulations at thin 1 and the
   variance 93%, while the rank mean sits at 45%, 40%, 47%, 23% and 33% over
   the three parameters together and at 7% to 23% on any one of them. log
   tau's pooled ranks are 2.18 and 2.19 in the end tenths and 0.64 to 0.83
   between, mean z +0.13 and variance z +13.9. So the answer to the project's
   question is the funnel's answer again: a chain from a prior start sits
   where it started, a study of such chains puts the truth at both ends
   evenly, and the one statistic that passes it is the one aimed at a shift.

2. The U is on all three parameters, not on the scale alone. mu's ends are
   2.32 and 2.31, theta_1's 2.20 and 2.18, the same as log tau's, and the
   band rejects every study on each of them alone at n = 100. A chain whose
   tau is stuck carries mu and the thetas with it, so the school effect does
   not mix where the scale does not, which I had the other way round.

3. Thinning shrinks the U and does not remove it. At thin 5 the ends are
   1.44 and 1.41 and the band still rejects 85% at 20 and 100% at 200; at
   thin 10, 1.26 and 1.24, 75% at 20 and 100% at 500. IACT(log tau) is 40 at
   the median and 82 at the 90th percentile over the prior studies, as on day
   1's real data, so thin 10 is a quarter of the lag.

4. The correct sampler fails the band at thin 1 too, on mu. 40% at 20 rising
   to 85% at 500, and on mu alone 45% at 100 and 82% at 500 against 7% and
   3% on log tau and theta_1; mu's ends are 1.19 and 1.24, a small U from a
   lag the band can see, as the walk's was on the sharper tests' day 1. Thin
   5 takes everything to 5% to 28% and thin 10 to 3% to 25%. So the textbook
   fix is correct and not independent, and a study of it at thin 1 is
   rejected by the sharpest omnibus test for the same reason the broken one
   is, only smaller.

5. The divergences are 2.62% of centred's proposals, 26 a simulation, and
   79.6% of its chains have one, against none non-centred. A count that
   fires on four chains in five is no use as a per-study flag when every
   study is rejected anyway; whether it sorts the studies at thin 10, where
   some pass at 20 simulations, is day 3's question. The nan guard caught
   0.5% of centred's chains.

6. One thing to carry: log tau's mean z on the correct sampler is -0.40,
   -0.38 and -0.43 at the three thins, which share chains, so that is one
   draw at 2.5 null sds pooled over 20,000 simulations, and the pooled
   histogram is flat, 0.97 to 1.05. A 1-in-80 draw or something small in the
   ranking of the scale; day 3 reruns the non-centred sampler on fresh
   studies and will say which.

Six predictions written before the run. Three right, one half, two wrong.

- Wrong: non-centred passes everything at thin 1, band and mean between 3%
  and 15% at every n. The band is 40% at 20 and 85% at 500, on mu.
- Right: centred at thin 1 is rejected by the band on over 80% at 50 and log
  tau's ranks are a U with both end tenths above 1.3. 100% at 20, 2.18 and
  2.19.
- Half: the rank mean passes centred at every n and thin, under 20%. 7% to
  23% on any one parameter, 12% to 47% on the three together.
- Right: thin 10 leaves centred's band above 80% at 500. 100%, and 90% at 50.
- Right: centred diverges on 1% to 3% of proposals over the prior studies and
  non-centred on none. 2.62% and 0.
- Wrong: theta_1's ranks alone pass centred under the band at thin 10, under
  30% at 500. 95%.

NumPy only. Fixed seeds. About 260 seconds.
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

# Each imported module keeps its own generator; reseed them all so this
# file's rows do not depend on what the earlier days drew.
es.RNG = np.random.default_rng(51)
d1.RNG = np.random.default_rng(52)
d2.RNG = np.random.default_rng(53)
RNG = np.random.default_rng(54)

L, SIZES = d1.L, d1.SIZES
EPS = 0.1
BURN = 100
PILOT = 990                      # transitions after burn, enough for L draws at thin 10
THINS = (1, 5, 10)
KEYS = ("chi2", "band", "mean", "var")
PARAMS = ("mu", "log tau", "theta_1")
KINDS = ("centred", "noncentred")
IACT_CHAINS = 200                # chains whose log tau pilot is kept for the IACT


def draw_studies(r):
    """r prior draws of the hierarchy and a data set from each. Returns the truth (r, 3) and y (r, J)."""
    mu = es.MU_SD * es.RNG.standard_normal(r)
    tau = es.TAU_SD * np.abs(es.RNG.standard_normal(r))
    theta = mu[:, None] + tau[:, None] * es.RNG.standard_normal((r, es.J))
    y = theta + es.SIGMA * es.RNG.standard_normal((r, es.J))
    return np.column_stack([mu, np.log(tau), theta[:, 0]]), y


def params(kind, z):
    """mu, log tau and theta_1 from either parameterisation's state."""
    return np.column_stack([z[:, 0], z[:, 1], es.thetas(kind, z)[:, 0]])


def sbc_pilot(kind, truth, y):
    """Burn, then PILOT transitions with the ranks of the truth accumulated at each thin.

    Returns ranks per thin (r, 3), the divergence count per chain over the
    pilot, and the log tau pilot of IACT_CHAINS chains at thin 1.
    """
    es.Y = y                     # day 1's potentials read the data from the module; (r, J) broadcasts per chain
    r = y.shape[0]
    z = es.prior_start(kind, r)
    for _ in range(BURN):
        z, _, _ = es.hmc(kind, z, EPS)
    below = {thin: np.zeros((r, 3), dtype=np.int32) for thin in THINS}
    div = np.zeros(r)
    kept = np.empty((PILOT, IACT_CHAINS))
    for k in range(1, PILOT + 1):
        z, _, d = es.hmc(kind, z, EPS)
        div += d
        kept[k - 1] = z[:IACT_CHAINS, 1]
        p = params(kind, z)
        for thin in THINS:
            if k % thin == 0 and k // thin <= L:
                below[thin] += p < truth
    return below, div, kept


def iact_summary(kept):
    """Median and 90% point of the IACT of log tau over the kept chains, the share the nan guard caught, and the lag-1."""
    tau = np.array([es.integrated_act(kept[:, c]) for c in range(kept.shape[1])])
    bad = ~np.isfinite(tau)
    finite = tau[~bad]
    xc = kept - kept.mean(0, keepdims=True)
    rho1 = (xc[:-1] * xc[1:]).mean() / (xc * xc).mean()
    return np.median(finite), np.quantile(finite, 0.9), bad.mean(), rho1


def shape(ranks, p, groups=10):
    """Pooled histogram of parameter p's ranks in tenths, count over expected, as the funnel days printed it."""
    g = ranks[:, :, p].ravel() * groups // (L + 1)
    return " ".join(f"{c:.2f}" for c in np.bincount(g, minlength=groups) / (g.size / groups))


def per_param(rr, stats, crit, key):
    """Share of studies rejected by one statistic on each parameter alone."""
    return [(stats[key](d1.rank_counts(rr[:, :, p])) > crit[key]).mean() for p in range(rr.shape[2])]


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
    sd_m, sd_v = d2.null_sds(nmax)

    truth, y = draw_studies(r)
    print(f"eight schools with data in it, {studies} studies x {nmax} simulations, eps {EPS}, "
          f"{es.STEPS} leapfrog steps, burn {BURN}, pilot {PILOT}, L = {L}")
    print(f"  the prior studies: tau < {es.NECK:.0f} on {100 * np.mean(np.exp(truth[:, 1]) < es.NECK):.1f}%, "
          f"median tau {np.median(np.exp(truth[:, 1])):.2f}")

    table, info, ranks = {}, {}, {}
    for kind in KINDS:
        below, div, kept = sbc_pilot(kind, truth, y)
        info[kind] = (div.sum() / (r * PILOT), (div > 0).mean(), div.mean()) + iact_summary(kept)
        ranks[kind] = {thin: below[thin].reshape(studies, nmax, 3) for thin in THINS}
        table[kind] = {thin: {n: d2.reject_any(ranks[kind][thin][:, :n], stats[n], crits[n]) for n in SIZES}
                       for thin in THINS}
        print(f"  {kind} done, {time.time() - t0:.0f} s", flush=True)

    print("\n1. the pilot: divergences per proposal, chains with one, per simulation; IACT of log tau; nan guard; lag-1")
    for kind, (share, any_div, per_sim, med, q90, bad, rho1) in info.items():
        print(f"  {kind:10s} div {100 * share:5.2f}% of proposals, {100 * any_div:5.1f}% of chains, "
              f"{per_sim:5.2f} a simulation  iact median {med:5.1f} 90% {q90:6.1f}  nan guard {bad:.4f}  "
              f"rho1 {rho1:+.2f}")

    print(f"\n2. share of {studies} studies rejected, any of the three parameters, " + " / ".join(KEYS))
    for kind in KINDS:
        for thin in THINS:
            cells = "  ".join(f"n={n}: " + "/".join(f"{table[kind][thin][n][k]:.2f}" for k in KEYS) for n in SIZES)
            print(f"  {kind:10s} thin {thin:2d} {cells}")

    print("\n3. simulations to reach 80% rejections, " + " / ".join(KEYS))
    for kind in KINDS:
        cells = "   ".join(f"thin {thin:2d} " + " ".join(f"{d2.first_size(table[kind][thin], k):>5s}" for k in KEYS)
                           for thin in THINS)
        print(f"  {kind:10s} {cells}")

    print(f"\n4. each parameter alone at n = 100 and {nmax}, band / mean, " + " / ".join(PARAMS))
    for kind in KINDS:
        for thin in THINS:
            cells = []
            for n in (100, nmax):
                rr = ranks[kind][thin][:, :n]
                b = per_param(rr, stats[n], crits[n], "band")
                m = per_param(rr, stats[n], crits[n], "mean")
                cells.append(f"n={n}: " + " ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m)))
            print(f"  {kind:10s} thin {thin:2d} " + "  ".join(cells))

    print(f"\n5. pooled rank histograms in tenths at n = {nmax}, and log tau's mean and variance z averaged over studies")
    for kind in KINDS:
        for thin in THINS:
            rr = ranks[kind][thin]
            c = d1.rank_counts(rr[:, :, 1])
            mz = ((d2.rank_mean(c) - L / 2) / sd_m).mean()
            vz = ((d2.rank_var(c) - d2.V_UNIF) / sd_v).mean()
            print(f"  {kind:10s} thin {thin:2d} mu      {shape(rr, 0)}")
            print(f"  {'':10s} {'':7s} log tau {shape(rr, 1)}   mean z {mz:+.2f} var z {vz:+.2f}")
            print(f"  {'':10s} {'':7s} theta_1 {shape(rr, 2)}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
