"""Day 4 - SBC on the funnel, where the RMHMC project's bug passed every diagnostic.

The RMHMC project's day 3 dropped the log det from the Fisher metric's H, and
the chain went to v ~ N(-9, 9) at the right chain's acceptance and IACT with no
divergences. Here that sampler, the correct one at eps 0.2 and 0.8, and centred
HMC at eps 0.2 are put through SBC, all as whole arrays of chains in NumPy,
20 leapfrog steps a transition. The funnel has no data, so the posterior is
the prior and SBC reduces to ranking one exact draw among a chain's draws,
which is the only test the funnel admits and the one the project never ran.
40 studies per row, ranks of v and x_1, burn 100, thin 1 and 5.

1. The diagnostics cannot tell the bug apart, as before. Over 50 chains of
   3000 transitions the log det run has acceptance 0.993 against 0.994 and
   IACT(v) 2.89 against 2.99, and 0 divergences on both. Its mean v is -9.04
   and its neck share 94.85%, which is visible only because the funnel's
   answer is known.

2. SBC catches it on every study at 20 simulations, 3,980 transitions a
   study at thin 1. v's histogram has 95% of its ranks in the top tenth, 9.53
   times uniform, because the truth sits above nearly every draw. x_1's is a
   U, 4.74 and 4.60 at the ends, the neck's x being too tight to contain it.

3. Centred HMC is the one the divergence count and SBC agree on, and its
   histogram is not the shape I predicted. 4.9% of its proposals diverge.
   SBC rejects 25% at 20, 90% at 100 and 100% at 500 at thin 1, and 100% at
   500 still at thin 5. v's histogram is a U, 1.69 and 1.83 at the ends and
   0.79 to 0.89 between, not a pile at the bottom. The chains start from
   prior draws, and a chain that starts in the neck stays there as surely as
   one in the mouth stays out of it, so both ends fill: day 3's slow sampler
   again, a chain that does not cross the truth in a run. Its pooled moments
   look fine, mean v 0.07 and neck 8.08%, Var(v) 9.58 the only hint, and
   IACT(v) is 16.

4. The correct sampler needs thin 5 to pass. At thin 1 it is rejected on 15%
   at 500 with v's ends at 1.07 and 1.05, a small U from an IACT of 3. At
   thin 5, 3% at 500 and flat to 0.03. At eps 0.8 it diverges on 14.7% of
   proposals and SBC rejects 12% at 500 at both thinnings with histograms
   flat to 0.06 and 0.03, about two standard errors over the size on 40
   studies. So the divergence count flags a sampler SBC finds close to
   correct, as the RMHMC project's day 2 already said from Var(v) 8.82: the
   rejections cost mixing and not the target.

Five predictions written before the run. Three right, two half.

- Half: Fisher RMHMC at eps 0.2 rejects at most 10% at 500 simulations. 15%
  at thin 1, 3% at thin 5.
- Right: with the log det off every study is rejected at 20 simulations, v's
  histogram all in the top tenth. 100%, 9.53 in the top tenth.
- Half: centred HMC is rejected on at least 80% at 500, v's histogram piled
  in the bottom tenth. 100%, but a U and not a pile.
- Right: the divergence count flags centred HMC and is 0 with the log det
  off. 4.9% and 0.
- Right: acceptance and IACT(v) of the log det run within 5% of the correct
  run's. 0.993 against 0.994, 2.89 against 2.99.

The cost against the divergence count: the count is free and blind to the
one bug that moves the target, and SBC caught that bug at 4,000 transitions,
about what a single chain diagnostic costs. What the count sees is a chain
that loses proposals, which SBC either agrees is broken, centred HMC, or
cannot separate from the size, eps 0.8.

NumPy only. Fixed seeds. About 175 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcmc-from-scratch"))
from day1_metropolis import integrated_act  # noqa: E402

RNG = np.random.default_rng(24)

# The RMHMC project's funnel: v ~ N(0, SIG^2), x_1..x_n | v ~ N(0, e^v), d = 3.
SIG, NX = 3.0, 2
GV = 1.0 / SIG**2 + 0.5 * NX       # Fisher mass on v, day 2 of that project
NECK = -4.16
L, BINS, BURN = 99, 20, 100
EPS, STEPS = 0.2, 20


def prior(r):
    v = SIG * RNG.standard_normal(r)
    return np.exp(0.5 * v)[:, None] * RNG.standard_normal((r, NX)), v


def rmhmc(x, v, eps=EPS, logdet=True):
    """The RMHMC project's day 2 step on whole arrays of chains, Fisher metric.

    G = diag(e^-v, .., e^-v, GV). With the log det in H it cancels the n v / 2
    in U. logdet=False leaves n v / 2 in H, which is that project's day 3 bug
    and makes the chain exact on p(x, v) sqrt(det G), v ~ N(-9, 9).
    Returns (x, v, accepted, diverged).
    """
    off = 0.0 if logdet else 0.5 * NX

    def dhdv(v, x, pp):
        return v / SIG**2 - 0.5 * np.exp(-v) * (x * x).sum(1) + 0.5 * pp * np.exp(v) + off

    def h(v, x, px, pv):
        return (0.5 * v * v / SIG**2 + 0.5 * np.exp(-v) * (x * x).sum(1)
                + 0.5 * np.exp(v) * (px * px).sum(1) + 0.5 * pv * pv / GV + off * v)

    px = np.exp(-0.5 * v)[:, None] * RNG.standard_normal(x.shape)
    pv = np.sqrt(GV) * RNG.standard_normal(v.size)
    h0 = h(v, x, px, pv)
    xn, vn = x.copy(), v.copy()
    for _ in range(STEPS):
        px = px - 0.5 * eps * np.exp(-vn)[:, None] * xn
        pp = (px * px).sum(1)
        pv = pv - 0.5 * eps * dhdv(vn, xn, pp)
        v_new = vn + eps * pv / GV
        xn = xn + 0.5 * eps * (np.exp(vn) + np.exp(v_new))[:, None] * px
        vn = v_new
        pv = pv - 0.5 * eps * dhdv(vn, xn, pp)
        px = px - 0.5 * eps * np.exp(-vn)[:, None] * xn
    return accept(x, v, xn, vn, h(vn, xn, px, pv) - h0)


def hmc(x, v, eps=EPS):
    """Unit-mass centred HMC on U = v^2 / (2 SIG^2) + e^-v |x|^2 / 2 + n v / 2."""
    def grad(x, v):
        ev = np.exp(-v)
        return ev[:, None] * x, v / SIG**2 - 0.5 * ev * (x * x).sum(1) + 0.5 * NX

    def u(x, v):
        return 0.5 * v * v / SIG**2 + 0.5 * np.exp(-v) * (x * x).sum(1) + 0.5 * NX * v

    px, pv = RNG.standard_normal(x.shape), RNG.standard_normal(v.size)
    h0 = u(x, v) + 0.5 * ((px * px).sum(1) + pv * pv)
    xn, vn = x.copy(), v.copy()
    gx, gv = grad(xn, vn)
    for _ in range(STEPS):
        px, pv = px - 0.5 * eps * gx, pv - 0.5 * eps * gv
        xn, vn = xn + eps * px, vn + eps * pv
        gx, gv = grad(xn, vn)
        px, pv = px - 0.5 * eps * gx, pv - 0.5 * eps * gv
    return accept(x, v, xn, vn, u(xn, vn) + 0.5 * ((px * px).sum(1) + pv * pv) - h0)


def accept(x, v, xn, vn, dh):
    """Metropolis on the energy error. A divergence is a dH past 1000 or not finite, counted once."""
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(RNG.random(v.size)) < -np.nan_to_num(dh, nan=np.inf))
    return np.where(acc[:, None], xn, x), np.where(acc, vn, v), acc, div


SAMPLERS = {
    "rmhmc":     lambda x, v: rmhmc(x, v),
    "rmhmc 0.8": lambda x, v: rmhmc(x, v, eps=0.8),
    "nologdet":  lambda x, v: rmhmc(x, v, logdet=False),
    "hmc":       lambda x, v: hmc(x, v),
}


def sbc_ranks(kind, studies, nsim, thin):
    """Ranks of a prior draw among L thinned draws, chain from an independent prior draw.

    There are no data, so the posterior is the prior and each simulation is a
    chain on the funnel scored against one exact draw. Ranks of v and of x_1.
    Returns ranks (studies, nsim, 2), acceptance and the divergence share.
    """
    step = SAMPLERS[kind]
    r = studies * nsim
    x0, v0 = prior(r)
    x, v = prior(r)
    acc = div = 0.0
    for _ in range(BURN):
        x, v, _, _ = step(x, v)
    below = np.zeros((r, 2), dtype=np.int32)
    for _ in range(L):
        for _ in range(thin):
            x, v, a, d = step(x, v)
            acc, div = acc + a.mean(), div + d.mean()
        below[:, 0] += v < v0
        below[:, 1] += x[:, 0] < x0[:, 0]
    n = L * thin
    return below.reshape(studies, nsim, 2), acc / n, div / n


def chi2(ranks):
    """Day 2's Pearson statistic of the ranks in BINS bins against uniform, along axis 1."""
    bins = ranks * BINS // (L + 1)
    e = ranks.shape[1] / BINS
    return sum(((bins == b).sum(1) - e) ** 2 / e for b in range(BINS))


CRIT = {}


def sbc_any(ranks, reps=20000):
    """Share of studies where either parameter passes the simulated 95% point at this nsim."""
    n = ranks.shape[1]
    if n not in CRIT:
        CRIT[n] = np.quantile(chi2(RNG.integers(0, L + 1, (reps, n, 1)))[:, 0], 0.95)
    return (chi2(ranks) > CRIT[n]).any(1).mean()


def shape(ranks, k, groups=10):
    """Pooled histogram of parameter k's ranks in tenths, as count over expected."""
    g = ranks[:, :, k].ravel() * groups // (L + 1)
    return " ".join(f"{c:.2f}" for c in np.bincount(g, minlength=groups) / (g.size / groups))


def chain_stats(kind, chains=50, n=3000):
    """What the RMHMC project looked at: mean v, Var(v), neck share, IACT(v), from p's draws."""
    step = SAMPLERS[kind]
    x, v = prior(chains)
    vs = np.empty((n, chains))
    for t in range(n):
        x, v, _, _ = step(x, v)
        vs[t] = v
    vs = vs[n // 10:]
    iact = np.median([integrated_act(vs[:, c]) for c in range(chains)])
    return vs.mean(), vs.var(), 100 * np.mean(vs < NECK), iact


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    studies = 40
    print(f"funnel d = {NX + 1}, sigma_v = {SIG}; eps {EPS}, {STEPS} leapfrog steps; L = {L}, "
          f"{BINS} bins, burn {BURN}; {studies} studies per row")

    print("\n1. the chain's own diagnostics, 50 chains x 3000 transitions from prior draws")
    print(f"  exact     mean v  0.00  Var(v)  9.00  neck  8.28%")
    for kind in SAMPLERS:
        m, var, neck, iact = chain_stats(kind)
        print(f"  {kind:9s} mean v {m:5.2f}  Var(v) {var:5.2f}  neck {neck:5.2f}%  IACT(v) {iact:6.2f}")

    print("\n2. sbc, thin 1 and thin 5, share of studies rejected by number of simulations")
    for thin in (1, 5):
        for kind in SAMPLERS:
            rk, acc, div = sbc_ranks(kind, studies, 500, thin)
            sb = [sbc_any(rk[:, :n]) for n in (20, 100, 500)]
            print(f"  thin {thin} {kind:9s} acc {acc:.3f} div {100 * div:5.2f}%   "
                  f"n=20 {sb[0]:.2f}  n=100 {sb[1]:.2f}  n=500 {sb[2]:.2f}   "
                  f"transitions/study at 500 {500 * (BURN + L * thin):,}")
            print(f"  {'':16s} v ranks  {shape(rk, 0)}")
            print(f"  {'':16s} x1 ranks {shape(rk, 1)}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
