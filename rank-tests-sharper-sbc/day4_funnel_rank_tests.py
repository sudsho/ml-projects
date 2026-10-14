"""Day 4 - The funnel again: centred HMC that failed for being slow against the log det bug that failed for being wrong.

SBC day 4 put the RMHMC project's funnel samplers through the binned
chi-squared at thin 1 and 5 and could not tell the two failures apart: the
chain with the log det dropped, which is exact on the wrong density, and
centred HMC, which leaves the right density invariant and does not cross the
truth in a run. Both were rejected on every study at 500 simulations. Here the
same three chains, the correct Fisher RMHMC at eps 0.2, the same with the log
det off and centred HMC at eps 0.2, each 20 leapfrog steps, are run as a
990-transition pilot after burn 100 and ranked five ways: every draw, every
fifth, every tenth, against an ESS-sized subsample of the pilot, and run on at
thin ceil(IACT) capped at 50 as on day 3. The ranks of v and of x_1 are then
put to the chi-squared, the band, the rank mean and the rank variance with
day 2's critical values, 40 studies a row, either parameter.

1. The log det bug is caught by everything at 20 simulations. Chi-squared,
   band, mean and variance all reject every study at 20 under every scheme,
   3,980 transitions a study at thin 1. v's ranks are 9.52 times uniform in
   the top tenth and x_1's a U at 4.63 and 4.70, as SBC day 4 saw, and
   running longer does not move either.

2. Centred HMC is the slow walk again and the rank mean passes it. At thin 1
   the band rejects it on 82% at 20 and 100% from 50, the variance 70% at 50
   and 97% at 100, the chi-squared 90% at 100, and the mean 20%, 17%, 10%,
   3% and 7% from 20 to 500, inside the size at every n and under every
   scheme. v's histogram is the U SBC day 4 drew, 1.72 and 1.81 at the ends
   and 0.78 to 0.85 between, and a U that moves the mean by nothing.

3. So on the funnel the two failures are told apart by the statistic and
   not by the thin. The wrong target is a pile at one end, the slow correct
   chain a U at both, the mean sees the first at 20 simulations and never
   the second. That is a fact about these two failures and not about slow
   against wrong in general: day 2's half kick is a hump, the mean cannot
   see it either, and a slow chain started from somewhere other than the
   prior would not be symmetric.

4. Thinning cannot do it at any price tried. HMC's pilot IACT is 16 at the
   median and 57 at the 90th percentile, thin 10 leaves the band at 90%
   from 50 and 100% from 200, and running on at thin ceil(IACT), capped at
   50, costs 3,257 transitions a simulation against 199, 16 times, and
   leaves the band at 95% at 50 and 100% at 200, with the U at 1.34 and
   1.32. The ESS subsample delays it, 28% at 20 and 57% at 100, and reaches
   100% at 500 as on day 3.

5. The correct RMHMC passes under every scheme. Band 15% to 20% from 100 on
   at thin 1, where SBC day 4's chi-squared had 15%, mean 3% to 10%, and the
   lag of 2.8 is a small U the band sees at the edge of the size, as the
   walk's was on day 1. Thin 5, thin 10, ess and longer all sit at 5% to 17%.

6. The nan guard fired on 3.4% of HMC's chains, and 12.6% hit the thin cap.
   4.62% of its proposals diverge, and a chain that starts in the neck can
   refuse every one of 990 proposals, which is a pilot with no variance.
   Day 3's guard gives it tau = 990 and the cap thins it by 50. HMC's lag 1
   on the funnel is +0.70, not the -0.66 of day 3's model, so the IACT
   floor that failed the correct HMC there did nothing here.

Six predictions written before the run. Three right, two half, one wrong.

- Right: the rank mean rejects the log det on every study at 20 under every
  scheme. 100%, and so did everything else.
- Half: at thin 1 the band and the variance take centred HMC to 100% by 50
  and the mean stays under 20% at 500. Band 100% at 50, variance 70% at 50
  and 97% at 100, mean 7%.
- Half: thin 10 leaves HMC's band over 50% at 500, and thin ceil(IACT)
  brings it under 20%. 100%, and 100%.
- Right: the correct RMHMC at thin 1 is rejected by the band on 20% to 30%
  at 500 and by the mean under 15%. 20% and 7%.
- Right: the ESS subsample delays HMC's band and still reaches 100% by 500.
  28% at 20, 100% at 500.
- Wrong: the nan guard fires on no chain and no thin hits the cap. 3.4% and
  12.6%, both on HMC.

NumPy only. Fixed seeds. About 245 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sampler-correctness-sbc"))
import day1_ecdf_bands as d1  # noqa: E402
import day2_shape_tests as d2  # noqa: E402
import day3_lag_aware_ranks as lag  # noqa: E402
import day3_planted_bugs as d3  # noqa: E402
import day4_funnel_sbc as fun  # noqa: E402

# Four modules each keep a generator and importing them reseeds one another.
# Reseed every one so this file's rows share no draws with any earlier day.
d3.RNG = np.random.default_rng(61)
d1.RNG = np.random.default_rng(62)
lag.RNG = np.random.default_rng(63)
fun.RNG = np.random.default_rng(64)
RNG = np.random.default_rng(65)

L, SIZES = d1.L, d1.SIZES
assert L == fun.L
PILOT = L * d3.THIN              # 990 transitions after burn, as on day 3
THIN_CAP = 50                    # the funnel's chains are longer-lagged than the model's; cap lower
KEYS = ("chi2", "band", "mean", "var")
KINDS = ("rmhmc", "nologdet", "hmc")


def pilot_chains(kind, r):
    """Burn 100, then PILOT transitions of v and x_1 kept at thin 1, from a prior start.

    The funnel has no data, so the truth is one more prior draw and the chain
    starts from an independent one, as SBC day 4 did. Returns the truth, the
    end state, the draws (r, PILOT, 2) and the divergence share over the pilot.
    """
    step = fun.SAMPLERS[kind]
    x0, v0 = fun.prior(r)
    x, v = fun.prior(r)
    for _ in range(fun.BURN):
        x, v, _, _ = step(x, v)
    draws = np.empty((r, PILOT, 2), dtype=np.float32)
    div = 0.0
    for k in range(PILOT):
        x, v, _, d = step(x, v)
        div += d.mean()
        draws[:, k, 0], draws[:, k, 1] = v, x[:, 0]
    return (v0, x0[:, 0]), (x, v), draws, div / PILOT


def ranks_thinned(truth, draws, thin):
    """Ranks of the truth among the first L draws taken every thin-th transition."""
    kept = draws[:, thin - 1::thin][:, :L]
    return np.stack([(kept[:, :, p] < truth[p][:, None]).sum(1) for p in range(2)], -1)


def ranks_ess(truth, draws, tau_hat):
    """Day 3's ESS-sized subsample ranks, on the funnel's pilot."""
    lag.RNG = RNG
    return lag.ranks_ess(truth, draws, tau_hat)


def ranks_longer(kind, truth, state, tau_hat):
    """Run each chain on from the pilot's end at thin ceil(tau_hat), capped, until it has L draws.

    Day 3's scheme with the funnel's (x, v) state. Returns ranks, transitions
    per chain and the thin used.
    """
    step = fun.SAMPLERS[kind]
    thin = np.minimum(np.ceil(tau_hat), THIN_CAP).astype(int)
    x, v = state[0].copy(), state[1].copy()
    below = np.zeros((v.size, 2), dtype=np.int64)
    for k in range(1, L * thin.max() + 1):
        act = np.nonzero(k <= L * thin)[0]
        xa, va, _, _ = step(x[act], v[act])
        x[act], v[act] = xa, va
        keep = act[k % thin[act] == 0]
        below[keep, 0] += v[keep] < truth[0][keep]
        below[keep, 1] += x[keep, 0] < truth[1][keep]
    return below, fun.BURN + PILOT + L * thin, thin


def lag1(x):
    """Lag-1 autocorrelation of each row of x, pooled over rows, the number day 3's HMC failure turned on."""
    xc = x - x.mean(1, keepdims=True)
    return (xc[:, :-1] * xc[:, 1:]).mean() / (xc * xc).mean()


def shape(ranks, k, groups=10):
    """Pooled histogram of parameter k's ranks in tenths, count over expected, as SBC day 4 printed it."""
    g = ranks[:, :, k].ravel() * groups // (L + 1)
    return " ".join(f"{c:.2f}" for c in np.bincount(g, minlength=groups) / (g.size / groups))


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    studies = 40
    nmax = SIZES[-1]

    stats, crits = {}, {}
    for n in SIZES:
        s, _ = d2.shape_stats(n)
        stats[n] = {k: s[k] for k in KEYS}
        crits[n] = {k: v for k, v in d2.critical_values(dict(s, n=n)).items() if k in KEYS}

    schemes = ("thin1", "thin5", "thin10", "ess", "longer")
    table, info, hist = {}, {}, {}
    for kind in KINDS:
        r = studies * nmax
        truth, state, draws, div = pilot_chains(kind, r)
        tau_hat, bad = lag.chain_iact(draws)
        rho1 = lag1(draws[:2000, :, 0].astype(np.float64))
        rk = {"thin1": ranks_thinned(truth, draws, 1), "thin5": ranks_thinned(truth, draws, 5),
              "thin10": ranks_thinned(truth, draws, 10), "ess": ranks_ess(truth, draws, tau_hat)}
        del draws
        rk["longer"], cost, thin = ranks_longer(kind, truth, state, tau_hat)
        info[kind] = (div, np.median(tau_hat), np.quantile(tau_hat, 0.9), rho1, bad.mean(),
                      (thin >= THIN_CAP).mean(), cost.mean())
        hist[kind] = {sc: (shape(rk[sc].reshape(studies, nmax, 2), 0),
                           shape(rk[sc].reshape(studies, nmax, 2), 1)) for sc in ("thin1", "longer")}
        table[kind] = {}
        for sc in schemes:
            rr = rk[sc].reshape(studies, nmax, 2)
            table[kind][sc] = {n: d2.reject_any(rr[:, :n], stats[n], crits[n]) for n in SIZES}
        print(f"  {kind} done, {time.time() - t0:.0f} s", flush=True)

    print(f"\nfunnel d = {fun.NX + 1}, sigma_v = {fun.SIG}; eps {fun.EPS}, {fun.STEPS} leapfrog steps; "
          f"L = {L}, burn {fun.BURN}, pilot {PILOT}; {studies} studies per row")

    print("\n1. the pilot: divergences, IACT (larger of v and x_1), lag-1 autocorrelation of v, cost of running longer")
    for kind, (div, med, q90, rho1, nan, capped, cost) in info.items():
        print(f"  {kind:8s} div {100 * div:5.2f}%  iact median {med:6.1f}  90% {q90:6.1f}  rho1(v) {rho1:+.2f}  "
              f"nan guard {nan:.4f}  at cap {capped:.3f}  transitions/sim {cost:7,.0f} against {fun.BURN + L:,} at thin 1")

    print(f"\n2. share of {studies} studies rejected, either parameter, " + " / ".join(KEYS))
    for kind in KINDS:
        for sc in schemes:
            cells = "  ".join(f"n={n}: " + "/".join(f"{table[kind][sc][n][k]:.2f}" for k in KEYS)
                              for n in (20, 50, 100, 200, 500))
            print(f"  {kind:8s} {sc:6s} {cells}")

    print("\n3. simulations to reach 80% rejections, " + " / ".join(KEYS))
    for kind in KINDS:
        cells = "   ".join(f"{sc} " + " ".join(f"{d2.first_size(table[kind][sc], k):>5s}" for k in KEYS)
                           for sc in schemes)
        print(f"  {kind:8s} {cells}")

    print("\n4. pooled rank histograms in tenths at n = 500, thin 1 and run longer")
    for kind in KINDS:
        for sc, (hv, hx) in hist[kind].items():
            print(f"  {kind:8s} {sc:6s} v  {hv}")
            print(f"  {'':8s} {'':6s} x1 {hx}")

    print("\n5. transitions a study to reach 80% on the two failures, thin 1 and run longer, " + " / ".join(KEYS))
    for kind in ("nologdet", "hmc"):
        for sc, per in (("thin1", fun.BURN + L), ("longer", info[kind][-1])):
            cells = []
            for k in KEYS:
                n = d2.first_size(table[kind][sc], k)
                cells.append(f"{int(n) * per:>11,.0f}" if not n.startswith(">") else f"{'>' + f'{nmax * per:,.0f}':>11s}")
            print(f"  {kind:8s} {sc:6s} " + " ".join(cells))

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
