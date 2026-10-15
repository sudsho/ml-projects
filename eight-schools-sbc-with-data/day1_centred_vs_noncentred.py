"""Day 1 - Eight schools, the centred and non-centred parameterisations, and where each diverges.

Every funnel row in the last two projects was a rank test against the prior,
because the funnel has no likelihood. Eight schools is the funnel with data in
it: eight estimated effects y_j with known standard errors, theta_j ~ N(mu,
tau^2), mu ~ N(0, 25), tau half-normal with sd 5 and sampled as u = log tau
with the Jacobian in the potential. The centred parameterisation carries the
thetas, the non-centred carries eta_j with theta_j = mu + tau eta_j, both as
whole arrays of chains in NumPy under the same unit-mass HMC with 20 leapfrog
steps. Both gradients are checked against central differences, the posterior
of log tau comes from a long non-centred run, and both samplers are run from
prior starts at four step sizes and read against it.

1. The gradients agree with central differences to 3e-11 relative on both
   potentials, so what follows is about the geometry and not a sign.

2. The reference, 50 chains x 20,000 after 2,000 burn at eps 0.1: acceptance
   0.996, no divergences, IACT(log tau) 2.1, the chain means within 0.01 of
   each other. log tau has mean 0.783 and sd 1.106, quantiles -1.34, 0.27,
   1.02, 1.56, 2.09, and 19.1% of the posterior has tau below 1. theta_1 is
   5.92 against its y of 28. I had the prior pulling tau higher than this;
   the half-normal at sd 5 and eight noisy schools leave a fifth of the mass
   in the neck.

3. The divergence count behaves as the textbook says. Centred HMC diverges on
   9.9% of proposals at eps 0.2, 1.07% at 0.1, 0.38% at 0.05 and 0.06% at
   0.02; the non-centred sampler on none at any of the four. The pooled
   moments do not. Centred's mean log tau is within 0.09 of the reference at
   every step size and its neck share within 2.5 points, which is not the
   bias I predicted.

4. What the divergences cost is mixing, and the pooled numbers hide it the
   way the funnel's U did on SBC day 4. IACT(log tau) is 30, 44, 92 and 239
   for centred against 1.9, 1.9, 7.3 and 40, and the sd of the 50 chain means
   is 0.78, 0.15, 0.21 and 0.49 against 0.03 to 0.12. The chains start from
   prior draws, a chain that starts with tau small stays there and one that
   starts with tau large stays out, and 50 of them pooled give the right mean
   because the starts were right. At eps 0.2 the 5% point of log tau is -2.13
   against -1.34, chains held in the neck by rejection, and at eps 0.1 it is
   -1.03, the tail missing. At eps 0.02, where the count is nearly silent,
   theta_1 comes out 4.96 against 5.92 with an IACT of 239: the step is small
   enough to stop diverging and too small to move.

5. So on this model the divergence count and the pooled posterior disagree
   in the other direction from the funnel's log det bug: there a chain with
   no divergences sat on the wrong target, here a chain with divergences at
   every step size reports the right pooled target and has not mixed. The
   rank tests on day 2 get both, and the question is which shape centred
   HMC's ranks take with data in the model.

Six predictions written before the run. Three right, one half, two wrong.

- Right: both gradients match central differences to 1e-6 relative. 3e-11.
- Right: centred HMC at eps 0.1 diverges on 1% to 5% of proposals, the
  non-centred under 0.1%. 1.07% and 0.
- Wrong: centred's mean log tau at eps 0.1 is at least 0.3 above the
  reference. +0.086, and -0.075 at eps 0.2.
- Right: IACT(log tau) under 10 non-centred and over 30 centred at eps 0.1.
  1.9 and 43.5.
- Wrong: the reference has mean log tau in [1.0, 1.5] and tau < 1 on 5% to
  15%. 0.78 and 19.1%.
- Half: eps 0.05 takes centred's divergences under 1% and leaves its neck
  share under half the reference's. 0.38%, and 18.6% against 19.1%.

NumPy only. Fixed seeds. About 45 seconds.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcmc-from-scratch"))
from day1_metropolis import integrated_act  # noqa: E402

RNG = np.random.default_rng(31)

# Rubin's eight schools: an estimated treatment effect and its standard error per school.
Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
J = Y.size
MU_SD = 5.0      # mu ~ N(0, MU_SD^2)
TAU_SD = 5.0     # tau ~ half-normal(TAU_SD), sampled as u = log tau with the Jacobian in U
STEPS = 20
NECK = 1.0       # the share of the posterior with tau below this is the neck

# A chain is a row of the array: column 0 is mu, 1 is u = log tau, 2.. are the eight thetas or etas.


def u_centred(z):
    """Potential of (mu, log tau, theta): -log p(mu) - log p(tau) - log |dtau/du| - log p(theta|mu,tau) - log p(y|theta)."""
    mu, u, th = z[:, 0], z[:, 1], z[:, 2:]
    tau2 = np.exp(2 * u)
    return (0.5 * mu * mu / MU_SD**2 + 0.5 * tau2 / TAU_SD**2 - u
            + 0.5 * ((th - mu[:, None]) ** 2).sum(1) / tau2 + J * u
            + 0.5 * (((Y - th) / SIGMA) ** 2).sum(1))


def grad_centred(z):
    mu, u, th = z[:, 0], z[:, 1], z[:, 2:]
    tau2 = np.exp(2 * u)
    dev = th - mu[:, None]
    g = np.empty_like(z)
    g[:, 0] = mu / MU_SD**2 - dev.sum(1) / tau2
    g[:, 1] = tau2 / TAU_SD**2 - 1 - (dev * dev).sum(1) / tau2 + J
    g[:, 2:] = dev / tau2[:, None] - (Y - th) / SIGMA**2
    return g


def u_noncentred(z):
    """Potential of (mu, log tau, eta) with theta = mu + tau eta: the hierarchy's scale moved into the likelihood."""
    mu, u, eta = z[:, 0], z[:, 1], z[:, 2:]
    tau = np.exp(u)
    th = mu[:, None] + tau[:, None] * eta
    return (0.5 * mu * mu / MU_SD**2 + 0.5 * tau * tau / TAU_SD**2 - u
            + 0.5 * (eta * eta).sum(1) + 0.5 * (((Y - th) / SIGMA) ** 2).sum(1))


def grad_noncentred(z):
    mu, u, eta = z[:, 0], z[:, 1], z[:, 2:]
    tau = np.exp(u)
    th = mu[:, None] + tau[:, None] * eta
    res = (th - Y) / SIGMA**2          # d/dtheta of the likelihood term
    g = np.empty_like(z)
    g[:, 0] = mu / MU_SD**2 + res.sum(1)
    g[:, 1] = tau * tau / TAU_SD**2 - 1 + tau * (res * eta).sum(1)
    g[:, 2:] = eta + res * tau[:, None]
    return g


def thetas(kind, z):
    """School effects from either parameterisation's state."""
    if kind == "centred":
        return z[:, 2:]
    return z[:, :1] + np.exp(z[:, 1:2]) * z[:, 2:]


MODELS = {"centred": (u_centred, grad_centred), "noncentred": (u_noncentred, grad_noncentred)}


def check_gradients(h=1e-5):
    """Central differences of each potential against its gradient at 20 prior-ish points."""
    z = np.concatenate([5 * RNG.standard_normal((20, 2)), 3 * RNG.standard_normal((20, J))], 1)
    for kind, (pot, grad) in MODELS.items():
        g = grad(z)
        fd = np.empty_like(g)
        for k in range(z.shape[1]):
            e = np.zeros(z.shape[1])
            e[k] = h
            fd[:, k] = (pot(z + e) - pot(z - e)) / (2 * h)
        rel = np.abs(fd - g).max() / np.abs(g).max()
        print(f"  {kind:10s} max |fd - grad| / max |grad| = {rel:.1e}")


def hmc(kind, z, eps):
    """Unit-mass HMC on whole arrays of chains, STEPS leapfrog steps. Returns (z, accepted, diverged)."""
    pot, grad = MODELS[kind]
    p = RNG.standard_normal(z.shape)
    h0 = pot(z) + 0.5 * (p * p).sum(1)
    zn = z.copy()
    g = grad(zn)
    for _ in range(STEPS):
        p = p - 0.5 * eps * g
        zn = zn + eps * p
        g = grad(zn)
        p = p - 0.5 * eps * g
    dh = pot(zn) + 0.5 * (p * p).sum(1) - h0
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(RNG.random(z.shape[0])) < -np.nan_to_num(dh, nan=np.inf))
    return np.where(acc[:, None], zn, z), acc, div


def prior_start(kind, chains):
    """Chains started from the prior: mu, tau, and thetas or etas drawn from it."""
    mu = MU_SD * RNG.standard_normal(chains)
    tau = TAU_SD * np.abs(RNG.standard_normal(chains))
    fine = RNG.standard_normal((chains, J))
    if kind == "centred":
        fine = mu[:, None] + tau[:, None] * fine
    return np.column_stack([mu, np.log(tau), fine])


def run(kind, eps, chains, n, burn):
    """Chains x n transitions after burn. Returns the log tau draws (n, chains), theta_1 draws, acceptance, divergence share."""
    z = prior_start(kind, chains)
    acc = div = 0.0
    us = np.empty((n, chains))
    t1 = np.empty((n, chains))
    for t in range(burn + n):
        z, a, d = hmc(kind, z, eps)
        if t >= burn:
            acc, div = acc + a.mean(), div + d.mean()
            us[t - burn] = z[:, 1]
            t1[t - burn] = thetas(kind, z)[:, 0]
    return us, t1, acc / n, div / n


def summarise(us, t1):
    """Mean, sd and 5% point of log tau, the neck share, mean theta_1, the median IACT of log tau, and the sd of the chain means."""
    iact = np.median([integrated_act(us[:, c]) for c in range(us.shape[1])])
    return us.mean(), us.std(), np.quantile(us, 0.05), 100 * np.mean(np.exp(us) < NECK), t1.mean(), iact, us.mean(0).std()


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    print(f"eight schools, mu ~ N(0, {MU_SD:.0f}^2), tau ~ half-normal({TAU_SD:.0f}), {STEPS} leapfrog steps a transition")

    print("\n1. gradients against central differences")
    check_gradients()

    print("\n2. reference: non-centred hmc, 50 chains x 20000 after 2000 burn, eps 0.1")
    us, t1, acc, div = run("noncentred", 0.1, 50, 20000, 2000)
    m, s, q05, neck, th1, iact, spread = summarise(us, t1)
    q = np.quantile(us, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  acc {acc:.3f} div {100 * div:.3f}%  log tau mean {m:.3f} sd {s:.3f}  "
          f"quantiles {' '.join(f'{v:.2f}' for v in q)}  tau < {NECK:.0f}: {neck:.2f}%  theta_1 {th1:.2f}  "
          f"IACT {iact:.1f}  chain-mean sd {spread:.3f}")
    ref = (m, q05, neck)

    print("\n3. both parameterisations at the same step sizes, 50 chains x 3000 after 500 burn")
    print(f"  reference: log tau mean {ref[0]:.3f} 5% {ref[1]:.2f}  tau < {NECK:.0f}: {ref[2]:.2f}%")
    for eps in (0.2, 0.1, 0.05, 0.02):
        for kind in MODELS:
            us, t1, acc, div = run(kind, eps, 50, 3000, 500)
            m, s, q05, neck, th1, iact, spread = summarise(us, t1)
            print(f"  eps {eps:.2f} {kind:10s} acc {acc:.3f} div {100 * div:6.3f}%  log tau mean {m:6.3f} "
                  f"({m - ref[0]:+.3f}) sd {s:.3f} 5% {q05:5.2f}  tau < {NECK:.0f}: {neck:5.2f}%  "
                  f"theta_1 {th1:5.2f}  IACT {iact:6.1f}  chain-mean sd {spread:.3f}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
