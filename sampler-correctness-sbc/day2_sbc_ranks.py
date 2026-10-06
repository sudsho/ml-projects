"""Day 2 - Simulation-based calibration from scratch on the same Gibbs sampler.

Geweke's test compares moments of the joint. SBC (Talts et al. 2018) compares
ranks: draw the truth from the prior, data from the model, run the sampler on
that data, and count how many of L posterior draws fall below the truth. For a
sampler that targets p(theta | y), the truth is one more draw from the same
posterior, so its rank among L draws is uniform on 0..L whatever the model.
Same model as day 1, same two bugs, L = 99 and 20 bins, 100 independent SBC
studies per row so each rejection rate is a rate and not one coin.

1. The thinning experiment had nothing to thin. Day 1's lag 1 of 0.85 in mu
   belonged to the successive-conditional chain, which draws a fresh y every
   sweep and so has to wander the whole joint. SBC holds y still, and the
   Gibbs chain on p(mu, tau | y) has lag 1 at 0.03 in mu and 0.07 in tau
   unthinned, about zero from thin 2. Unthinned it rejects at 2% on mu, 3% on
   tau and 5% for either, and the mu histogram is flat to 0.01. Thin 5 came
   out at 10% and 9%, 17% for either, with nothing in the lag to explain it.
   Three reruns on other seeds gave 1% to 7%, so that row is a bad draw of
   100 studies, about 2 se out, and a 100-study rate should not be read to
   better than 3 points either way. The chain thinning is for is day 1's slow
   Metropolis step, which day 3 runs here: thin 10 rejects on every study and
   thin 100 at 12%.

2. The chi-squared critical value is calibrated by simulating exactly uniform
   ranks at the same number of simulations, and it is 30.0 at both 50 and
   1000, against chi-squared(19)'s 30.1. Two and a half ranks to a bin was
   expected to break the approximation and does not.

3. The sd bug is caught by 50 simulations on every study. mu's posterior is
   too narrow by sqrt(prec), about 2.9, so the truth falls outside the draws
   and the histogram is a U, 3.05 in each end tenth and 0.36 in the middle.
   It leaks into tau too, 23% at 200 and 83% at 1000, because mu held tight
   to its mean shrinks the sum of squares and lifts tau's draws, which shows
   as a tau histogram tilting down from 1.33 to 0.79.

4. The shape bug is the expensive one. It lowers tau's draws, the truth ranks
   high and tau's histogram climbs from 0.62 to 1.43, but it takes 200
   simulations to reach 50% and 500 to reach 95%. At thin 1 that is 500 x 199
   = 99,500 sweeps per study. Day 1's joint test caught the same bug on every
   replication at M = 1000, 1000 sweeps and 1000 independent joint draws, so
   on this bug Geweke is about a hundred times cheaper than SBC.

Six predictions written before the run. Three right, one half, two wrong.

- Right: correct sampler at thin 10, each parameter rejects at 3% to 7%. 3%
  and 5%.
- Wrong: unthinned, mu rejects on more than half the studies. 2%, because the
  0.85 was the joint chain's and not the posterior chain's.
- Wrong: thin 5 is what brings mu under 7%. Thin 1 already is, and the thin 5
  row itself came out at 10%.
- Right: the sd bug is caught on at least 90% at 100 simulations. 100% at 50.
- Half: the shape bug needs at least 500 simulations for 80%. 52% at 200 and
  95% at 500, so the 80% point is somewhere between and the bound is loose.
- Right: the sd bug's mu histogram is a U and the shape bug's tau histogram
  piles at the top.

NumPy only. Fixed seeds. About 110 seconds.
"""

import time

import numpy as np

RNG = np.random.default_rng(22)

# Day 1's model and sampler, unchanged. mu ~ N(M0, 1 / K0), tau ~ Gamma(A0,
# rate B0), y_1..y_N | mu, tau ~ N(mu, 1 / tau).
M0, K0, A0, B0, N = 0.0, 1.0, 3.0, 2.0, 5

L = 99      # posterior draws per simulation, so the rank takes 100 values
BINS = 20   # five ranks to a bin
BURN = 100


def prior(r):
    mu = M0 + RNG.standard_normal(r) / np.sqrt(K0)
    tau = RNG.gamma(A0, 1.0 / B0, r)
    return mu, tau


def simulate_data(mu, tau):
    return mu[:, None] + RNG.standard_normal((mu.size, N)) / np.sqrt(tau)[:, None]


def gibbs_sweep(mu, tau, ysum, y, bug=None):
    """Day 1's sweep with y fixed, which SBC holds still while the chain runs.

    'shape' uses A0 + (N - 1) / 2 and 'sd' draws mu with sd 1 / prec.
    """
    prec = K0 + N * tau
    mean = (K0 * M0 + tau * ysum) / prec
    sd = 1.0 / prec if bug == "sd" else 1.0 / np.sqrt(prec)
    mu = mean + sd * RNG.standard_normal(mu.size)
    ss = ((y - mu[:, None]) ** 2).sum(1)
    shape = A0 + ((N - 1) / 2 if bug == "shape" else N / 2)
    return mu, RNG.gamma(shape, 1.0 / (B0 + ss / 2))


def sbc_ranks(studies, nsim, thin, bug=None):
    """Ranks of the prior draw among L thinned posterior draws, for studies x nsim simulations.

    The chain starts from a second, independent prior draw rather than at the
    truth, so a short burn-in would show up as bias and not be hidden by it.
    Draws are counted on the fly, nothing is stored. Returns (studies, nsim, 2).
    """
    r = studies * nsim
    mu0, tau0 = prior(r)
    y = simulate_data(mu0, tau0)
    ysum = y.sum(1)
    mu, tau = prior(r)
    for _ in range(BURN):
        mu, tau = gibbs_sweep(mu, tau, ysum, y, bug)
    below = np.zeros((r, 2), dtype=np.int32)
    for _ in range(L):
        for _ in range(thin):
            mu, tau = gibbs_sweep(mu, tau, ysum, y, bug)
        below[:, 0] += mu < mu0
        below[:, 1] += tau < tau0
    return below.reshape(studies, nsim, 2)


def chi2_stat(ranks):
    """Pearson chi-squared of the binned ranks against uniform, along axis 1."""
    nsim = ranks.shape[1]
    bins = ranks * BINS // (L + 1)
    counts = np.stack([(bins == b).sum(1) for b in range(BINS)], 1)
    expected = nsim / BINS
    return ((counts - expected) ** 2 / expected).sum(1), counts


CRIT = {}


def critical(nsim, reps=20000):
    """The 95% point of the statistic under exactly uniform ranks at this nsim.

    Calibrated by simulation instead of read off chi-squared(19), so at 50
    simulations, 2.5 to a bin, the test's size is still 5% for a perfect
    sampler and any excess belongs to the sampler.
    """
    if nsim not in CRIT:
        u = RNG.integers(0, L + 1, (reps, nsim))
        stat, _ = chi2_stat(u[:, :, None])
        CRIT[nsim] = np.quantile(stat[:, 0], 0.95)
    return CRIT[nsim]


def reject(ranks):
    stat, _ = chi2_stat(ranks)
    hit = stat > critical(ranks.shape[1])
    return hit.mean(0), hit.any(1).mean()


def shape_line(ranks, k, groups=10):
    """Pooled rank histogram of parameter k in `groups` bins, as count over expected."""
    g = ranks[:, :, k].ravel() * groups // (L + 1)
    c = np.bincount(g, minlength=groups) / (g.size / groups)
    return " ".join(f"{x:.2f}" for x in c)


def lag1(thin, r=2000, m=4000):
    """Lag-1 autocorrelation of the thinned chain in mu and tau at a fixed y per chain."""
    mu0, tau0 = prior(r)
    y = simulate_data(mu0, tau0)
    ysum = y.sum(1)
    mu, tau = mu0, tau0
    xs = np.empty((m, r, 2))
    for j in range(m):
        for _ in range(thin):
            mu, tau = gibbs_sweep(mu, tau, ysum, y)
        xs[j, :, 0], xs[j, :, 1] = mu, tau
    x = xs - xs.mean(0)
    return ((x[1:] * x[:-1]).mean(0) / (x * x).mean(0)).mean(0)


def main():
    t0 = time.time()
    studies = 100
    print(f"L = {L}, {BINS} bins, burn {BURN}; {studies} SBC studies per row; "
          f"crit at 50/1000 sims {critical(50):.1f} / {critical(1000):.1f} vs chi2(19) 30.1")

    print("\n1. correct sampler, 1000 simulations, rejection against thinning")
    for thin in (1, 2, 5, 10, 20):
        ac = lag1(thin)
        rk = sbc_ranks(studies, 1000, thin)
        per, anyp = reject(rk)
        cost = 1000 * (BURN + L * thin)
        print(f"  thin {thin:2d}  lag1 mu={ac[0]:.3f} tau={ac[1]:.3f}  "
              f"reject mu={per[0]:.2f} tau={per[1]:.2f} any={anyp:.2f}  sweeps/study={cost:,}")
        if thin in (1, 10):
            print(f"           mu histogram  {shape_line(rk, 0)}")

    print("\n2. power against day 1's bugs, thin 10, by number of simulations")
    hists = {}
    for bug in ("sd", "shape"):
        for nsim in (50, 100, 200, 500, 1000):
            rk = sbc_ranks(studies, nsim, 10, bug)
            per, anyp = reject(rk)
            print(f"  {bug:5s} nsim {nsim:4d}: reject mu={per[0]:.2f} tau={per[1]:.2f} any={anyp:.2f}")
            hists[bug] = rk
    for bug, rk in hists.items():
        print(f"  {bug:5s} histograms at 1000: mu  {shape_line(rk, 0)}")
        print(f"  {'':5s}                    tau {shape_line(rk, 1)}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
