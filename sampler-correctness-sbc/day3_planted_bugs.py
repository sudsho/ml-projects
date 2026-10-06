"""Day 3 - The planted bugs, each put to Geweke's test and to SBC at matched cost.

Days 1 and 2 planted their bugs in a Gibbs sweep. This puts the plan's bugs
in the samplers they belong to, on the same model moved in (mu, log tau):
unit-mass HMC with the last half kick left out, an RMHMC whose metric for mu
is the conditional precision K0 + N tau with the log det term dropped, a Gibbs
step for mu and a multiplicative random walk for tau without its Hastings
factor, and day 1's slow sampler. Three correct samplers, HMC, RMHMC with the
log det and the walk with the factor, set the size. Geweke at M = 1000 and
10000 on 100 replications, SBC at 100 and 500 simulations on 40 studies.

1. The size row is the same for all three correct samplers and for day 1's
   Gibbs. Geweke rejects 28% to 31% at M = 1000 and 16% to 19% at 10000,
   SBC 3% to 10% at 100 and 3% to 7% at 500. A Geweke row at M = 1000 is
   read against 30%, not 5%. At section 1's y all three give E[tau] 0.961 or
   0.962, and quadrature with mu integrated out gives 0.9612, so the 0.961
   the bugs are measured against is the posterior's and not the samplers'.

2. The missing log det is Geweke's bug and costs SBC 500 simulations. It moves
   E[tau] at one y from 0.961 to 1.037, 8%, at the right chain's acceptance,
   0.993. Geweke rejects on every replication at M = 1000. SBC sees the tilt,
   tau's histogram falling from 1.42 to 0.71, but rejects on 23% at 100
   simulations and 75% at 500, 545,000 transitions a study against 1,000.

3. The missing Hastings factor is caught by both. A whole unit off tau's shape
   moves E[tau] 19%, to 0.775. Geweke 100% at M = 1000, SBC 80% at 100 and
   100% at 500, the histogram climbing from 0.34 to 2.03, day 2's shape bug
   in the same direction at about three times the slope.

4. The dropped half kick is nearly invisible. Acceptance falls from 0.988 to
   0.906 and E[tau] from 0.961 to 0.953, under 1%. Geweke rejects 24% at
   M = 1000, inside the size, and 52% at 10000, SBC 17% at 500. Its histogram
   is the only hump in the table, 0.86 and 0.92 at the ends and 1.02 to 1.06
   between: the truth sits inside the draws too often, so the chain is too
   wide rather than shifted.

5. The slow sampler fails SBC as it failed Geweke, and the start is not why.
   At burn 100 and thin 10 SBC rejects 88% at 100 and 100% at 500, and tau's
   histogram is all ends, 1.81 and 1.74 with 0.75 to 0.90 between: in a run
   of 99 draws the chain often never crosses the truth. Burn 2000 changes
   nothing, 80% and 100%, ends 1.87 and 1.73. Thin 100 brings it to 17% and
   12%, ends 1.08 and 1.03, still over the size at 11,900 transitions a
   simulation. Day 2 found nothing to thin on a Gibbs chain. This is the
   chain thinning is for, and neither test tells it apart from a bug.

Six predictions written before the run. One right, two half, three wrong.

- Right: HMC and RMHMC reject at most 25% under Geweke at M = 10000 and 10%
  under SBC at 500. 16% and 19%, 5% and 7%.
- Wrong: the dropped half kick is caught by Geweke on 90% at M = 1000. 24%.
- Half: the missing log det is caught by both on 90% at their smallest cost.
  Geweke 100% at M = 1000, SBC 23% at 100.
- Half: the missing Hastings factor likewise. Geweke 100%, SBC 80% at 100.
- Wrong: the slow sampler passes SBC at thin 10. 100% at 500.
- Wrong: the half kick's tau histogram tilts one way. It is a hump.

NumPy only. Fixed seeds. About 165 seconds.
"""

import time

import numpy as np

RNG = np.random.default_rng(23)

# Days 1 and 2's model. mu ~ N(M0, 1 / K0), tau ~ Gamma(A0, rate B0),
# y_1..y_N | mu, tau ~ N(mu, 1 / tau). The samplers below move (mu, s) with
# s = log tau, where the target picks up the Jacobian and has shape A0 + N / 2.
M0, K0, A0, B0, N = 0.0, 1.0, 3.0, 2.0, 5

EPS, STEPS = 0.15, 8   # leapfrog step and steps per HMC transition
L, BINS, BURN, THIN = 99, 20, 100, 10


def prior(r):
    mu = M0 + RNG.standard_normal(r) / np.sqrt(K0)
    tau = RNG.gamma(A0, 1.0 / B0, r)
    return mu, tau


def simulate_data(mu, tau):
    return mu[:, None] + RNG.standard_normal((mu.size, N)) / np.sqrt(tau)[:, None]


def potential(mu, s, y):
    """U = -log p(mu, s | y) up to a constant, and its gradient in (mu, s)."""
    tau = np.exp(s)
    ss = ((y - mu[:, None]) ** 2).sum(1)
    u = 0.5 * K0 * (mu - M0) ** 2 - (A0 + N / 2) * s + tau * (B0 + ss / 2)
    du_mu = K0 * (mu - M0) - tau * (y.sum(1) - N * mu)
    du_s = -(A0 + N / 2) + tau * (B0 + ss / 2)
    return u, du_mu, du_s


def hmc(mu, tau, y, halfstep_bug=False):
    """Unit-mass HMC on (mu, log tau). The bug leaves out the last half kick."""
    s = np.log(tau)
    pm, ps = RNG.standard_normal(mu.size), RNG.standard_normal(mu.size)
    u0, gm, gs = potential(mu, s, y)
    h0 = u0 + 0.5 * (pm ** 2 + ps ** 2)
    m, q = mu.copy(), s.copy()
    for k in range(STEPS):
        pm, ps = pm - 0.5 * EPS * gm, ps - 0.5 * EPS * gs
        m, q = m + EPS * pm, q + EPS * ps
        u1, gm, gs = potential(m, q, y)
        if not (halfstep_bug and k == STEPS - 1):
            pm, ps = pm - 0.5 * EPS * gm, ps - 0.5 * EPS * gs
    h1 = u1 + 0.5 * (pm ** 2 + ps ** 2)
    acc = np.log(RNG.random(mu.size)) < h0 - h1
    return np.where(acc, m, mu), np.exp(np.where(acc, q, s))


def rmhmc(mu, tau, y, logdet=True):
    """Generalised leapfrog with mu's metric G(s) = K0 + N e^s, the conditional precision.

    G depends on s only, so both implicit solves of the RMHMC project's day 1
    have closed form here and the step is explicit. H = U + 0.5 log G +
    0.5 pm^2 / G + 0.5 ps^2; logdet=False drops the 0.5 log G from H and from
    its gradient together, the RMHMC project's day 3 bug, which turns the
    chain into exact HMC on p(mu, s) sqrt(G(s)).
    """
    c = 1.0 if logdet else 0.0
    s = np.log(tau)

    def G(q):
        return K0 + N * np.exp(q)

    def dh_ds(q, gs, pm):
        g, dg = G(q), N * np.exp(q)
        return gs + 0.5 * c * dg / g - 0.5 * pm ** 2 * dg / g ** 2

    pm = np.sqrt(G(s)) * RNG.standard_normal(mu.size)
    ps = RNG.standard_normal(mu.size)
    u0, gm, gs = potential(mu, s, y)
    h0 = u0 + 0.5 * c * np.log(G(s)) + 0.5 * pm ** 2 / G(s) + 0.5 * ps ** 2
    m, q = mu.copy(), s.copy()
    for _ in range(STEPS):
        pm = pm - 0.5 * EPS * gm
        ps = ps - 0.5 * EPS * dh_ds(q, gs, pm)
        q_new = q + EPS * ps
        m = m + 0.5 * EPS * pm * (1 / G(q) + 1 / G(q_new))
        q = q_new
        _, gm, gs = potential(m, q, y)
        ps = ps - 0.5 * EPS * dh_ds(q, gs, pm)   # both final kicks read the half-step pm
        pm = pm - 0.5 * EPS * gm
    u1 = potential(m, q, y)[0]
    h1 = u1 + 0.5 * c * np.log(G(q)) + 0.5 * pm ** 2 / G(q) + 0.5 * ps ** 2
    acc = np.log(RNG.random(mu.size)) < h0 - h1
    return np.where(acc, m, mu), np.exp(np.where(acc, q, s))


def gibbs_mh(mu, tau, y, step, hastings=True):
    """Gibbs for mu, then tau' = tau e^(step z) accepted on p(tau'|.) / p(tau|.).

    The multiplicative proposal is not symmetric in tau and needs the factor
    tau' / tau. hastings=False leaves it out, so the chain targets p(tau | .) /
    tau, which is the shape bug of day 1 with a whole unit off instead of half.
    """
    prec = K0 + N * tau
    mu = (K0 * M0 + tau * y.sum(1)) / prec + RNG.standard_normal(mu.size) / np.sqrt(prec)
    ss = ((y - mu[:, None]) ** 2).sum(1)
    prop = tau * np.exp(step * RNG.standard_normal(tau.size))
    logr = (A0 + N / 2 - 1) * np.log(prop / tau) - (prop - tau) * (B0 + ss / 2)
    if hastings:
        logr += np.log(prop / tau)
    acc = np.log(RNG.random(tau.size)) < logr
    return mu, np.where(acc, prop, tau)


SAMPLERS = {
    "hmc":      lambda mu, tau, y: hmc(mu, tau, y),
    "rmhmc":    lambda mu, tau, y: rmhmc(mu, tau, y),
    "mh":       lambda mu, tau, y: gibbs_mh(mu, tau, y, 0.5),
    "halfstep": lambda mu, tau, y: hmc(mu, tau, y, halfstep_bug=True),
    "nologdet": lambda mu, tau, y: rmhmc(mu, tau, y, logdet=False),
    "hastings": lambda mu, tau, y: gibbs_mh(mu, tau, y, 0.5, hastings=False),
    "slow":     lambda mu, tau, y: gibbs_mh(mu, tau, y, 0.05),
}


def geweke_any(kind, r, m, batches=50):
    """Day 1's joint test, five functions, batch-means se; share of r replications rejected."""
    step = SAMPLERS[kind]

    def g(mu, tau, y):
        return np.stack([mu, tau, mu ** 2, tau ** 2, mu * y.mean(1)], -1)

    mc = np.empty((r, m, 5))
    for j in range(m):
        mu, tau = prior(r)
        mc[:, j] = g(mu, tau, simulate_data(mu, tau))
    sc = np.empty((r, m, 5))
    mu, tau = prior(r)
    y = simulate_data(mu, tau)
    for j in range(m):
        mu, tau = step(mu, tau, y)
        y = simulate_data(mu, tau)
        sc[:, j] = g(mu, tau, y)
    b = m // batches
    v_sc = sc.reshape(r, batches, b, 5).mean(2).var(1, ddof=1) / batches
    z = (mc.mean(1) - sc.mean(1)) / np.sqrt(mc.var(1, ddof=1) / m + v_sc)
    return (np.abs(z) > 1.96).any(1).mean()


def sbc_ranks(kind, studies, nsim, burn=BURN, thin=THIN):
    """Day 2's ranks, chain started from an independent prior draw, (studies, nsim, 2)."""
    step = SAMPLERS[kind]
    r = studies * nsim
    mu0, tau0 = prior(r)
    y = simulate_data(mu0, tau0)
    mu, tau = prior(r)
    for _ in range(burn):
        mu, tau = step(mu, tau, y)
    below = np.zeros((r, 2), dtype=np.int32)
    for _ in range(L):
        for _ in range(thin):
            mu, tau = step(mu, tau, y)
        below[:, 0] += mu < mu0
        below[:, 1] += tau < tau0
    return below.reshape(studies, nsim, 2)


def chi2(ranks):
    """Day 2's Pearson statistic of the ranks in BINS bins against uniform, along axis 1."""
    bins = ranks * BINS // (L + 1)
    e = ranks.shape[1] / BINS
    return sum(((bins == b).sum(1) - e) ** 2 / e for b in range(BINS))


def sbc_any(ranks, reps=20000):
    """Share of studies where either parameter's chi-squared passes day 2's simulated 95% point."""
    u = RNG.integers(0, L + 1, (reps, ranks.shape[1], 1))
    crit = np.quantile(chi2(u)[:, 0], 0.95)
    return (chi2(ranks) > crit).any(1).mean()


def exact_tau_mean(y):
    """E[tau | y] by quadrature on tau, with mu integrated out in closed form.

    y | tau ~ N(M0, I / tau + 11' / K0), whose determinant and inverse have
    closed forms in the two eigenvalues 1 / tau and 1 / tau + N / K0. Uses no
    random numbers, so it leaves every seeded row below as it was.
    """
    t = np.linspace(1e-4, 8.0, 400001)
    a, b, r = 1.0 / t, 1.0 / K0, y - M0
    logdet = (N - 1) * np.log(a) + np.log(a + N * b)
    quad = (r @ r) / a - b * r.sum() ** 2 / (a * (a + N * b))
    lp = (A0 - 1) * np.log(t) - B0 * t - 0.5 * logdet - 0.5 * quad
    w = np.exp(lp - lp.max())
    return (t * w).sum() / w.sum()


def tau_shape(ranks, groups=10):
    """Pooled histogram of tau's ranks in tenths, as count over expected.

    A tilt is a shift, a U is draws too narrow or a chain that never crosses
    the truth in a run, and a hump is draws too wide.
    """
    g = ranks[:, :, 1].ravel() * groups // (L + 1)
    return " ".join(f"{x:.2f}" for x in np.bincount(g, minlength=groups) / (g.size / groups))


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    r, studies = 100, 40
    sbc_cost = BURN + L * THIN
    print(f"eps {EPS}, {STEPS} leapfrog steps; geweke {r} replications, sbc {studies} studies, "
          f"thin {THIN} ({sbc_cost} transitions per simulation)")

    print("\n1. acceptance and the stationary E[tau] each sampler reaches at one fixed y")
    mu0, tau0 = prior(1)
    y = np.repeat(simulate_data(mu0, tau0), 4000, 0)
    print(f"  exact    E[tau | y] {exact_tau_mean(y[0]):.4f} by quadrature")
    for kind, step in SAMPLERS.items():
        mu, tau = prior(4000)
        for _ in range(300):
            mu, tau = step(mu, tau, y)
        last = tau.copy()
        acc, means = 0.0, []
        for _ in range(200):
            mu, tau = step(mu, tau, y)
            acc += (tau != last).mean() / 200
            last = tau.copy()
            means.append(tau.mean())
        print(f"  {kind:8s} accept {acc:.3f}  E[tau] {np.mean(means):.3f}")

    print("\n2. share of replications or studies rejected, and the cost in transitions")
    for kind in SAMPLERS:
        gw = [geweke_any(kind, r, m) for m in (1000, 10000)]
        rk = sbc_ranks(kind, studies, 500)
        sb = [sbc_any(rk[:, :n]) for n in (100, 500)]
        print(f"  {kind:8s} geweke M=1000 {gw[0]:.2f}  M=10000 {gw[1]:.2f}   "
              f"sbc n=100 {sb[0]:.2f}  n=500 {sb[1]:.2f}   tau ranks {tau_shape(rk)}")
    print(f"  per replication: geweke 1,000 / 10,000 transitions, "
          f"sbc {100 * sbc_cost:,} / {500 * sbc_cost:,}")

    print("\n3. the slow sampler again: is it the start or the lag")
    for burn, thin in ((2000, 10), (2000, 100)):
        rk = sbc_ranks("slow", studies, 500, burn, thin)
        sb = [sbc_any(rk[:, :n]) for n in (100, 500)]
        print(f"  burn {burn} thin {thin:3d}  sbc n=100 {sb[0]:.2f}  n=500 {sb[1]:.2f}   "
              f"tau ranks {tau_shape(rk)}   transitions/sim {burn + L * thin:,}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
