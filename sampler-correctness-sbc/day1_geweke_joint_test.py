"""Day 1 - Geweke's joint-distribution test on a conjugate Gibbs sampler.

The RMHMC project's day 3 chain dropped the log det term and passed acceptance,
IACT and the divergence count exactly as the right chain did. Those are
statistics of the chain. Geweke (2004) asks the sampler a question from outside
it: draw (theta, y) from the joint two ways, independently from prior and
model, and by a chain that alternates one sampler sweep given y with a fresh y
given theta. A sampler that leaves p(theta | y) invariant makes the joint the
second chain's stationary law, so the two sets of moments must agree, and the
test is a z-score on their difference. The target is the normal model with
unknown mean and precision, mu ~ N(0, 1), tau ~ Gamma(3, 2), five data points,
where the Gibbs conditionals are closed form and each bug is one wrong symbol.

1. Size depends on the standard error more than on the test. The successive
   chain is autocorrelated, lag 1 at 0.85 for mu and 0.38 for tau, so an iid
   standard error rejects a correct sampler 46% of the time on mu and 70% on
   any of the five functions. Batch means with 50 batches fix that at M =
   10000, 4.5% to 7.0% per function and 18% for any, but not at M = 1000,
   where batches of 20 sweeps are too short and it is 8.5% to 10% and 29%.

2. Two of the bugs are caught at the smallest M. tau's shape A0 + (n - 1) / 2,
   the sample-variance n - 1, moves E[tau] by 0.29 sd and is rejected on every
   replication at M = 1000 with median max |z| of 5.1. mu drawn with sd 1 /
   prec where it should be 1 / sqrt(prec) is caught as often, but only by the
   second moments: E[mu] does not move (-0.0002 sd) and the mu column rejects
   at 6% to 8%, which is the size. A test of first moments alone passes it.

3. Dropping the prior's rate B0 from tau's conditional is not caught, because
   there is nothing left to test. With no floor on the rate, a large tau pulls
   the next y onto mu, the sum of squares shrinks and tau grows, and all 200
   chains overflow to inf in 69 to 195 sweeps, median 94. Every z is nan and
   a harness counting |z| > 1.96 counts zero rejections at every M.

4. A correct sampler that is slow fails. tau by random-walk Metropolis on
   log tau, step 0.05, leaves the posterior invariant and has lag-1
   autocorrelation 0.995 on tau. It is rejected on 86% of replications at
   M = 1000, 44% at 10000 and still 20% at 50000, where tau alone is at 10%.
   The batches are shorter than the chain's memory, so the test reads slow as
   wrong, which is the case day 3 has to separate from a bug. It could not:
   SBC rejects the same sampler on every study at 500 simulations and thin 10,
   with burn 100 or 2000, and only thin 100 brings it near the size.

Six predictions written before the run. Two right, one half, three wrong.

- Half: on the correct sampler with batch means each function rejects at 3%
  to 7% and any of five at most 25%. True at M = 10000, 4.5% to 7.0% and 18%,
  and not at M = 1000, 8.5% to 10% and 29%.
- Right: the naive standard error rejects mu above 10% at M = 10000. 46.5%.
- Right: the sd bug is caught on at least 90% of replications at M = 1000.
  100%.
- Wrong: the shape bug needs more than M = 1000, under 50% there and 80% by
  10000. It is 100% at 1000.
- Wrong: the rate bug is caught on at least 90% at M = 1000. The chain dies
  before it has a mean and the count is 0.
- Wrong: the slow sampler's rejection stays at the size. 86% at M = 1000.

NumPy only. Fixed seeds. About 45 seconds.
"""

import time

import numpy as np

RNG = np.random.default_rng(21)

# Semi-conjugate normal model. mu ~ N(M0, 1 / K0), tau ~ Gamma(A0, rate B0),
# y_1..y_N | mu, tau ~ N(mu, 1 / tau). The two full conditionals are closed
# form, so a Gibbs sweep is two draws and every bug below is one wrong symbol.
M0, K0, A0, B0, N = 0.0, 1.0, 3.0, 2.0, 5


def prior(r):
    mu = M0 + RNG.standard_normal(r) / np.sqrt(K0)
    tau = RNG.gamma(A0, 1.0 / B0, r)
    return mu, tau


def simulate_data(mu, tau):
    return mu[:, None] + RNG.standard_normal((mu.size, N)) / np.sqrt(tau)[:, None]


def gibbs_sweep(mu, tau, y, bug=None):
    """One sweep, mu then tau, vectorised over independent replications.

    bug names a single wrong symbol:
      'shape'  tau's shape A0 + (N - 1) / 2, the n - 1 from a sample variance
      'rate'   tau's rate drops the prior's B0
      'sd'     mu drawn with sd 1 / prec instead of 1 / sqrt(prec)
      'slow'   no bug: tau by random-walk Metropolis on log tau, step 0.05
    """
    prec = K0 + N * tau
    mean = (K0 * M0 + tau * y.sum(1)) / prec
    sd = 1.0 / prec if bug == "sd" else 1.0 / np.sqrt(prec)
    mu = mean + sd * RNG.standard_normal(mu.size)
    ss = ((y - mu[:, None]) ** 2).sum(1)
    if bug == "slow":
        def logp(t):
            return (A0 + N / 2) * np.log(t) - t * (B0 + ss / 2)  # incl. log-jacobian
        prop = tau * np.exp(0.05 * RNG.standard_normal(tau.size))
        acc = np.log(RNG.random(tau.size)) < logp(prop) - logp(tau)
        return mu, np.where(acc, prop, tau)
    shape = A0 + ((N - 1) / 2 if bug == "shape" else N / 2)
    rate = (0.0 if bug == "rate" else B0) + ss / 2
    return mu, RNG.gamma(shape, 1.0 / rate)


def test_functions(mu, tau, y):
    """The moments compared: first and second of each parameter, and a cross term with the data."""
    return np.stack([mu, tau, mu ** 2, tau ** 2, mu * y.mean(1)], -1)


G_NAMES = ["mu", "tau", "mu^2", "tau^2", "mu*ybar"]


def marginal_conditional(r, m):
    """m independent draws of (theta, y) from the joint, for r replications at once."""
    out = np.empty((r, m, len(G_NAMES)))
    for j in range(m):
        mu, tau = prior(r)
        out[:, j] = test_functions(mu, tau, simulate_data(mu, tau))
    return out


def successive_conditional(r, m, bug=None):
    """theta from the prior once, then alternate a sampler sweep given y and a fresh y given theta.

    If the sweep leaves p(theta | y) invariant, the joint p(theta, y) is the
    stationary law of this chain, so its moments must match the independent
    draws'. A bug changes the stationary law and the moments move with it.
    """
    mu, tau = prior(r)
    y = simulate_data(mu, tau)
    out = np.empty((r, m, len(G_NAMES)))
    for j in range(m):
        mu, tau = gibbs_sweep(mu, tau, y, bug)
        y = simulate_data(mu, tau)
        out[:, j] = test_functions(mu, tau, y)
    return out


def batch_var_of_mean(x, batches=50):
    """Variance of the mean of x along axis 1 by batch means, for autocorrelated chains.

    Biased low whenever a batch is shorter than the chain's memory. At M = 1000
    a batch is 20 sweeps, short enough to lift the Gibbs chain's size to 8.5%
    to 10% at lag-1 0.85 and far too short for the slow sampler's 0.995. Both
    chains start in the joint, so the mean is unbiased and the excess
    rejections are the standard error's.
    """
    r, m, k = x.shape
    b = m // batches
    means = x[:, : b * batches].reshape(r, batches, b, k).mean(2)
    return means.var(1, ddof=1) / batches


def geweke_z(mc, sc, naive=False):
    v_sc = sc.var(1, ddof=1) / sc.shape[1] if naive else batch_var_of_mean(sc)
    v_mc = mc.var(1, ddof=1) / mc.shape[1]
    return (mc.mean(1) - sc.mean(1)) / np.sqrt(v_mc + v_sc)


def reject_rates(z):
    """Per-function share of replications with |z| > 1.96, and the share with any of the five."""
    hit = np.abs(z) > 1.96
    return hit.mean(0), hit.any(1).mean()


def main():
    t0 = time.time()
    r = 200
    print(f"model: mu ~ N({M0}, 1/{K0}), tau ~ Gamma({A0}, {B0}), n = {N}; {r} replications per row")

    print("\n1. size on the correct sampler, batch-means and naive standard errors")
    for m in (1000, 10000):
        mc, sc = marginal_conditional(r, m), successive_conditional(r, m)
        for naive in (False, True):
            per, anyg = reject_rates(geweke_z(mc, sc, naive))
            tag = "naive" if naive else "batch"
            print(f"  M = {m:6d} {tag}: " + " ".join(f"{g}={p:.3f}" for g, p in zip(G_NAMES, per))
                  + f"  any={anyg:.3f}")

    print("\n2. lag-1 autocorrelation of each g on the successive chain, M = 10000")
    for name, chain in (("gibbs", sc), ("slow", successive_conditional(r, 10000, "slow"))):
        ac = []
        for k in range(len(G_NAMES)):
            x = chain[:, :, k] - chain[:, :, k].mean(1, keepdims=True)
            ac.append(((x[:, 1:] * x[:, :-1]).mean(1) / (x * x).mean(1)).mean())
        print(f"  {name:5s} " + " ".join(f"{g}={a:.3f}" for g, a in zip(G_NAMES, ac)))

    print("\n3. power against each bug, batch-means se, share of replications rejected")
    power = {}
    np.seterr(all="ignore")
    for bug in ("shape", "rate", "sd", "slow"):
        for m in (1000, 10000, 50000):
            rr = r if m < 50000 else 50
            mc, sc = marginal_conditional(rr, m), successive_conditional(rr, m, bug)
            per, anyg = reject_rates(geweke_z(mc, sc))
            power[bug, m] = anyg
            z = geweke_z(mc, sc)
            print(f"  {bug:5s} M = {m:6d}: " + " ".join(f"{g}={p:.3f}" for g, p in zip(G_NAMES, per))
                  + f"  any={anyg:.3f}  median|z|max={np.median(np.abs(z).max(1)):.2f}")

    print("\n4. the rate bug's successive chain: sweeps until tau stops being a number")
    with np.errstate(all="ignore"):
        mu, tau = prior(r)
        y = simulate_data(mu, tau)
        dead = np.full(r, -1)
        for j in range(2000):
            mu, tau = gibbs_sweep(mu, tau, y, "rate")
            y = simulate_data(mu, tau)
            fresh = (dead < 0) & ~(np.isfinite(tau) & (tau < 1e300))
            dead[fresh] = j + 1
    died = dead[dead > 0]
    print(f"  {died.size} of {r} dead by 2000 sweeps, sweeps to death median {np.median(died):.0f}, "
          f"range {died.min()} to {died.max()}")

    print("\n5. stationary moment shift each bug causes, from the M = 50000 rows")
    for bug in ("shape", "sd"):
        mc, sc = marginal_conditional(50, 50000), successive_conditional(50, 50000, bug)
        d = sc.mean((0, 1)) - mc.mean((0, 1))
        rel = d / mc.std((0, 1))
        print(f"  {bug:5s}: " + " ".join(f"{g}={x:+.4f}" for g, x in zip(G_NAMES, rel)) + "  (in sd of g)")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
