"""Day 3 - Ranks with the autocorrelation taken out, two ways, on the same seven samplers.

Days 1 and 2 ranked the truth among every tenth draw of 990, whatever the
chain's lag. Here each chain keeps all 990 draws after burn 100 as a pilot,
its IACT is measured from them with Sokal's window, the larger of mu's and
tau's, and the ranks are taken two ways. "ess" spends nothing more: the truth
is ranked among M = min(99, 990 / IACT) evenly spaced pilot draws and the rank
jittered back onto 0..99. "longer" keeps running the chain from the pilot's
end at thin ceil(IACT), capped at 200, until it has 99 draws, so a slow chain
pays for its lag in transitions. Chi-squared, band and rank mean, critical
values from day 2, 40 studies, either parameter, size about 10% with a
standard error near 5.

1. The pilot's IACT is 1.0 for HMC and the half kick, 2.4 for both RMHMCs,
   6.7 and 7.5 for the two walks and 77 for the slow walk, 122 at its 90th
   percentile, none at the cap. Thinning by it costs the slow walk 8,997
   transitions a simulation against thin 10's 1,090, 8.3 times, and every
   other sampler 1,189 to 1,960.

2. Running longer does most of the job on the slow walk and not all of it.
   The band falls from 100% at 20 simulations to 15% to 20% up to 200 and
   42% at 500, the chi-squared to 7% to 17%. The 990-draw pilot reads a lag
   it cannot hold, so thin 77 at the median is still short, as SBC day 3's
   thin 100 left 12% to 17%. The rank mean was already at the size there,
   7% to 25% under thin 10, and stays there.

3. Ranking against an ESS-sized subsample does not. It delays the band on
   the slow walk from 20 simulations to 200 and the chi-squared from 100 to
   200, and by 500 both reject every study. A subsample of the same pilot
   holds the same lag; ten evenly spaced draws are no less correlated with
   each other than ninety-nine.

4. The bugs are still caught. The log det reaches 95% to 97% under the band
   and 100% under the mean at 500 by all three schemes, the Hastings bug 97%
   under the mean at 50 by all three. Neither scheme costs the shift bugs
   anything, and neither helps them.

5. The new failure is HMC. Its lag-1 autocorrelation is -0.66 on both
   parameters, the draws are antithetic, the IACT estimate is floored at 1,
   and the run-longer scheme thins it by 1. The band then rejects the correct
   HMC on 17% to 28% of studies where thin 10 gave 0% to 10%, and the half
   kick, lag-1 -0.41, on 45% at 500. So half the half kick's best number
   yet is the correct sampler's own anti-correlation, and the IACT floor that
   keeps a thin from being under 1 is what lets it through.

6. The nan guard fired on 2 of 20,000 chains, both under the dropped log
   det, where a pilot had no variance or no finite estimate. Without it they would have had a
   nan IACT and a thin of 1.

Six predictions written before the run. Two right, one half, three wrong.

- Wrong: the three correct samplers hold 10% or under under both schemes
  at every n. HMC at 17% to 28% on the band when run longer, RMHMC 17% under
  the ess ranks.
- Wrong: running longer passes the slow walk, band under 15% at 500. 42%.
- Right: the ess ranks do not pass the slow walk, band over 50% at 500. 100%.
- Right: the log det still fails under the run-longer ranks, rank mean 80%
  by 500. 100%.
- Wrong: thinning by IACT costs the slow walk over 10 times thin 10's
  transitions. 8.3 times, and it was not enough.
- Half: the nan guard fires on under 1% of chains, all on the slow walk.
  0.01%, and on the dropped log det.

Day 4 has a sharper target than the plan gave it. The rank mean already
passed the slow walk under plain thin 10, so the cheapest test there is a
choice of statistic and not a choice of thin. On the funnel the question is
whether centred HMC's lag reads as bias the same way, and whether the
log det bug still shows up as a shift the rank mean sees.

NumPy only. Fixed seeds. About 160 seconds.
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
import day3_planted_bugs as d3  # noqa: E402

# Importing day 2 reseeds day 1's and day 3's generators; reseed all three so
# this file's rows share no draws with either earlier day.
d3.RNG = np.random.default_rng(51)
d1.RNG = np.random.default_rng(52)
RNG = np.random.default_rng(53)

L, SIZES = d1.L, d1.SIZES
PILOT = L * d3.THIN          # 990 transitions after burn, what thin 10 already pays for
THIN_CAP = 200               # longest thin the run-longer scheme is allowed
KEYS = ("chi2", "band", "mean")


def iact(x, c=5.0):
    """Integrated autocorrelation time of each row of x, Sokal's window, and a nan flag.

    tau = 1 + 2 sum_{k=1}^{W} rho_k with the smallest W >= c tau(W). A row whose
    draws are all equal has no autocorrelation to read, rho is 0 / 0, and the
    estimate would be nan and every comparison with it False, which thins such
    a chain by 1. The guard gives those rows tau = T, as if the whole run were
    one draw, and returns which rows it fired on.
    """
    t = x.shape[1]
    xc = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(xc, 2 * t, axis=1)
    acf = np.fft.irfft(f * np.conj(f), axis=1)[:, :t]
    dead = acf[:, 0] <= 0
    rho = acf / np.where(dead, 1.0, acf[:, 0])[:, None]
    taus = 1 + 2 * np.cumsum(rho[:, 1:], 1)
    w = np.arange(1, t)
    ok = w[None, :] >= c * taus
    first = np.where(ok.any(1), ok.argmax(1), t - 2)
    tau = np.maximum(taus[np.arange(x.shape[0]), first], 1.0)
    bad = dead | ~np.isfinite(tau)
    return np.where(bad, float(t), tau), bad


def chain_iact(draws, chunk=2000):
    """Larger of mu's and tau's IACT for each chain of draws (r, T, 2), chunked for memory."""
    r = draws.shape[0]
    tau, bad = np.empty(r), np.zeros(r, dtype=bool)
    for a in range(0, r, chunk):
        sl = slice(a, a + chunk)
        tm, bm = iact(draws[sl, :, 0].astype(np.float64))
        tt, bt = iact(draws[sl, :, 1].astype(np.float64))
        tau[sl], bad[sl] = np.maximum(tm, tt), bm | bt
    return tau, bad


def pilot_chains(kind, r):
    """Burn 100, then PILOT transitions kept at thin 1. Returns truth, data, end state, draws."""
    step = d3.SAMPLERS[kind]
    mu0, tau0 = d3.prior(r)
    y = d3.simulate_data(mu0, tau0)
    mu, tau = d3.prior(r)
    for _ in range(d3.BURN):
        mu, tau = step(mu, tau, y)
    draws = np.empty((r, PILOT, 2), dtype=np.float32)
    for k in range(PILOT):
        mu, tau = step(mu, tau, y)
        draws[:, k, 0], draws[:, k, 1] = mu, tau
    return (mu0, tau0), y, (mu, tau), draws


def ranks_thin10(truth, draws):
    """Day 1 and 2's ranks: every tenth pilot draw, 99 of them."""
    kept = draws[:, d3.THIN - 1::d3.THIN]
    return np.stack([(kept[:, :, p] < truth[p][:, None]).sum(1) for p in range(2)], -1)


def ranks_ess(truth, draws, tau_hat):
    """Rank against M = min(L, T / tau_hat) evenly spaced pilot draws, put back on 0..L.

    With M varying by chain the ranks are on different scales, so each is
    jittered to u = (r + V) / (M + 1), uniform on (0, 1) when the M draws are
    independent of each other and of the truth, then cut into L + 1 cells.
    No transitions beyond the pilot.
    """
    r = draws.shape[0]
    m = np.clip(np.floor(PILOT / tau_hat), 1, L).astype(int)
    out = np.empty((r, 2), dtype=np.int64)
    for mm in np.unique(m):
        rows = np.nonzero(m == mm)[0]
        idx = ((np.arange(mm) + 0.5) * PILOT / mm).astype(int)
        sub = draws[rows][:, idx]
        for p in range(2):
            below = (sub[:, :, p] < truth[p][rows, None]).sum(1)
            u = (below + RNG.random(rows.size)) / (mm + 1)
            out[rows, p] = np.minimum((u * (L + 1)).astype(int), L)
    return out


def ranks_longer(kind, truth, y, state, tau_hat):
    """Keep running each chain from the pilot's end, thin ceil(tau_hat) capped at THIN_CAP, L draws.

    Only the chains still short of their L draws are stepped, so the cost of a
    chain is PILOT + L * its own thin. Returns ranks and transitions per chain.
    """
    step = d3.SAMPLERS[kind]
    thin = np.minimum(np.ceil(tau_hat), THIN_CAP).astype(int)
    mu, tau = state[0].copy(), state[1].copy()
    below = np.zeros((mu.size, 2), dtype=np.int64)
    for k in range(1, L * thin.max() + 1):
        act = np.nonzero(k <= L * thin)[0]
        mu[act], tau[act] = step(mu[act], tau[act], y[act])
        keep = act[k % thin[act] == 0]
        below[keep, 0] += mu[keep] < truth[0][keep]
        below[keep, 1] += tau[keep] < truth[1][keep]
    return below, d3.BURN + PILOT + L * thin, thin


def reject_any(ranks, stats, crit):
    """Share of studies where either parameter's statistic passes its 95% point, as on day 2."""
    return d2.reject_any(ranks, stats, crit)


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

    schemes = ("thin10", "ess", "longer")
    table, info = {}, {}
    for kind in d3.SAMPLERS:
        r = studies * nmax
        truth, y, state, draws = pilot_chains(kind, r)
        tau_hat, bad = chain_iact(draws)
        rk = {"thin10": ranks_thin10(truth, draws), "ess": ranks_ess(truth, draws, tau_hat)}
        del draws
        rk["longer"], cost, thin = ranks_longer(kind, truth, y, state, tau_hat)
        info[kind] = (np.median(tau_hat), np.quantile(tau_hat, 0.9), bad.mean(),
                      (thin >= THIN_CAP).mean(), cost.mean())
        table[kind] = {}
        for sc in schemes:
            rr = rk[sc].reshape(studies, nmax, 2)
            table[kind][sc] = {n: reject_any(rr[:, :n], stats[n], crits[n]) for n in SIZES}
        print(f"  {kind} done, {time.time() - t0:.0f} s", flush=True)

    print("\n1. the measured IACT from the 990-draw pilot, larger of mu and tau, and what thinning by it costs")
    for kind, (med, q90, nan, capped, cost) in info.items():
        print(f"  {kind:8s} iact median {med:7.1f}  90% {q90:7.1f}   nan guard {nan:.4f}   "
              f"at cap {capped:.3f}   transitions/sim {cost:8,.0f} against {d3.BURN + PILOT:,}")

    print(f"\n2. share of {studies} studies rejected, either parameter, " + " / ".join(KEYS))
    for kind in d3.SAMPLERS:
        for sc in schemes:
            cells = "  ".join(f"n={n}: " + "/".join(f"{table[kind][sc][n][k]:.2f}" for k in KEYS)
                              for n in (50, 100, 200, 500))
            print(f"  {kind:8s} {sc:6s} {cells}")

    print("\n3. simulations to reach 80% rejections, " + " / ".join(KEYS))
    for kind in d3.SAMPLERS:
        cells = "   ".join(f"{sc} " + " ".join(f"{d2.first_size(table[kind][sc], k):>5s}" for k in KEYS)
                           for sc in schemes)
        print(f"  {kind:8s} {cells}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
