"""Day 4 - Step size and the neck: both samplers over eps, the cost of a passing study, and the funnel's metric on eight schools.

Day 3 left the non-centred sampler passing at eps 0.2 and failing on either
side of it, at 0.1 from a trajectory too short for mu and at 0.4 from
rejections, and centred HMC failing at every thin tried at eps 0.1. This file
asks three things. Whether it is the trajectory's length or the step count
that passes mu: eps 0.2 with 10 steps is the same two units as eps 0.1 with
20 at half the cost, eps 0.1 with 40 steps is four units at double, and eps
0.3 with 20 sits between the pass and the break. Whether centred HMC passes
at any step size, eps 0.05 and 0.2 at thin 1 and 5 beside day 2's 0.1. And
whether the RMHMC project's lesson carries: the funnel's metric was the
negative Hessian averaged over x given v, a function of v alone, which made
the generalised leapfrog explicit. On the centred eight schools the same
average gives masses 1/25 + 8/tau^2 on mu and 1/tau^2 + 1/sigma_j^2 on each
theta_j, functions of u = log tau alone; the one cross term that does not
average out, mu against theta_j at -1/tau^2, is dropped, and u's own mass is
held at 2J, the hierarchy's Fisher information for log tau, with the prior's
2 tau^2 / 25 dropped so the step stays explicit. The log determinant goes
into the Hamiltonian and p is drawn from N(0, G(u)). The cost of a passing
study is counted in gradient evaluations a simulation: thin x steps x 99.

1. It is the trajectory, not the step count. eps 0.2 with 10 steps, two
   units at half the cost, fails the band on mu as eps 0.1 with 20 did on
   day 3, 93% at 500 against 97%, ends 1.25 and 1.25; eps 0.1 with 40
   steps, four units at double the cost, passes everything at thin 1, 20%
   at 500. So the pass at eps 0.2 bought on day 3 was the trajectory's
   length, and the same four units cost 3960 gradients a simulation at
   eps 0.1 and 1980 at eps 0.2.

2. There is no window. eps 0.3 with 20 steps, six units, has acceptance
   0.915 against 0.98 at 0.2, divergences on 0.003% of proposals, and
   fails the band on 100% of studies at 20 simulations. Not on mu, which
   passes, and only 30% on log tau at 500: on the school effects, eta_1
   a U with ends 2.45 and 2.56, theta_1 1.54 and 1.57. eta has unit mass
   and a near unit-normal posterior, so its trajectories turn once every
   2 pi = 6.28 units, and six units bring eta nearly back to where it
   started. That is a guess from the period and not tested here; what is
   measured is that the break at 0.3 is on eta and not a rejection U.

3. Centred HMC passes at no step size. At eps 0.05 the band rejects 100%
   at 20 simulations and at thin 5 still 100% at 500, log tau's ends 2.71
   and 3.02 at thin 1 and 1.70 and 2.10 at thin 5; at eps 0.2 the same
   100% with ends 1.95 and 1.85 and 1.48 and 1.41. The smaller step
   diverges less, 1.08% against 5.73% of proposals, and mixes worse, IACT
   49 against 19, so the U is deeper where the count is quieter, as day 1
   said it would be.

4. The funnel's metric removes the divergences and not the lag. RMHMC on
   the centred parameterisation diverges on none of the proposals at eps
   0.1, 0.2 and 0.4, acceptance 0.998, 0.989 and 0.960, IACT(log tau) 10,
   4.8 and 4.7 against centred HMC's 19 to 49. At thin 1 the band still
   rejects 100% at 500 at eps 0.1 and 0.2 and 62% at 0.4, log tau alone
   100%, 100% and 55%, a U tilted to the low end (1.69 and 1.39 at 0.1,
   1.14 and 1.03 at 0.4). At eps 0.4 thin 5 it passes, 12% at 500 and 5%
   on log tau. So the centred parameterisation passes for the first time
   in this project, at 9900 gradients a simulation, five times the
   non-centred sampler's 1980 at its day 3 eps 0.2.

5. What the metric does buy, and what it does not. The derived eta_1
   ranks are flat under RMHMC at every eps, end tenths 0.20, where centred
   HMC's were 0.26 to 0.37. The lag sits on log tau, not on mu as I
   expected from mu's mass 1/25 + 8/tau^2: at eps 0.4 thin 1 the band per
   parameter at 500 is 7% on mu, 55% on log tau, 10% on theta_1 and 5% on
   eta_1. u's mass is held at 2J, the Fisher information, while the prior
   term was dropped, so in the neck the metric is right for the thetas and
   too heavy for log tau, which is the next thing to try.

Six predictions written before the run. Three right, two half, one wrong.

- Right: eps 0.2 with 10 steps fails the band on mu above 70% at 500. 93%.
- Right: eps 0.1 with 40 steps passes everything at thin 1 under 25% at
  500. 20%.
- Wrong: eps 0.3 with 20 steps has acceptance between 0.8 and 0.95, no
  divergences, and passes under 25% at 500. 0.915 and 0.003%, near
  enough, but the band rejects 100% at 20, on eta.
- Right: centred HMC fails the band at 100% at 500 at eps 0.05 and 0.2 at
  thin 1 and thin 5, the U deeper at 0.05. 100% everywhere, log tau ends
  2.71 and 3.02 against 1.95 and 1.85.
- Half: RMHMC at eps 0.2 diverges under 0.5% and the band on log tau alone
  is under 50% at 500 at thin 1. No divergences at all, but log tau 100%.
- Half: RMHMC passes at thin 5 and not at thin 1, so centred costs under
  9900 and more than 1980 gradients. Passes at thin 5 and not thin 1 at
  eps 0.4, but at 9900 exactly, since thin 2 to 4 were not run.

NumPy only. Fixed seeds. About 650 seconds.
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
from day3_eta_ranks_and_divergences import KEYS, PARAMS, draw_studies, params  # noqa: E402

# Day 3's module reseeds the imported generators when it loads; reseed again
# so these studies share nothing with the 16th's or the 18th's.
es.RNG = np.random.default_rng(71)
d1.RNG = np.random.default_rng(72)
d2.RNG = np.random.default_rng(73)
RNG = np.random.default_rng(74)

L, SIZES = d1.L, d1.SIZES
BURN = 100
PILOT = 495                      # enough for L draws at thin 5
IACT_CHAINS = 200
J = es.J
G_U = 2.0 * J                    # the hierarchy's Fisher information for log tau, held constant
B_MU = float(J)                  # mu's mass is 1/MU_SD^2 + J e^-2u
A = np.concatenate([[1.0 / es.MU_SD**2], 1.0 / es.SIGMA**2])      # the constant part of the masses of mu and the thetas
B = np.concatenate([[B_MU], np.ones(J)])                          # the e^-2u part

# (kind, sampler, eps, steps, thins, pilot)
TRAJECTORY = [("noncentred", "hmc", 0.2, 10), ("noncentred", "hmc", 0.1, 40), ("noncentred", "hmc", 0.3, 20)]
CENTRED_EPS = (0.05, 0.2)
RMHMC_EPS = (0.1, 0.2, 0.4)
RMHMC_THIN_EPS = 0.4             # the pilot run at thin 1 and 5


# ---------------------------------------------------------------- the metric

def masses(u):
    """G(u) for mu and the thetas, (r, J + 1): the averaged negative Hessian, a function of log tau alone."""
    return A + B * np.exp(-2.0 * u)[:, None]


def dlogdet_du(u):
    """d/du of 0.5 log det G, with u's constant mass contributing nothing."""
    e = np.exp(-2.0 * u)[:, None]
    return (-B * e / (A + B * e)).sum(1)


def dkin_du(u, pq):
    """d/du of 0.5 sum p_k^2 / G_k(u) over mu and the thetas, pq (r, J + 1)."""
    e = np.exp(-2.0 * u)[:, None]
    g = A + B * e
    return (pq * pq * B * e / (g * g)).sum(1)


def hamiltonian(z, p):
    g = masses(z[:, 1])
    pq = np.concatenate([p[:, :1], p[:, 2:]], 1)
    return (es.u_centred(z) + 0.5 * np.log(g).sum(1)
            + 0.5 * (pq * pq / g).sum(1) + 0.5 * p[:, 1] ** 2 / G_U)


def rmhmc(z, eps, steps):
    """RMHMC on the centred parameterisation with G(u) diagonal, generalised leapfrog made explicit.

    mu's and the thetas' momentum half steps do not contain p because G does not
    depend on them; u's contains only theirs, by then known, and not p_u because
    its mass is constant; u's position step is explicit for the same reason, and
    the others' average G^-1 at the two known u's. Returns (z, accepted, diverged).
    """
    r = z.shape[0]
    g0 = masses(z[:, 1])
    p = np.empty_like(z)
    p[:, 0] = np.sqrt(g0[:, 0]) * es.RNG.standard_normal(r)
    p[:, 1] = np.sqrt(G_U) * es.RNG.standard_normal(r)
    p[:, 2:] = np.sqrt(g0[:, 1:]) * es.RNG.standard_normal((r, J))
    h0 = hamiltonian(z, p)
    zn, pn = z.copy(), p.copy()
    with np.errstate(all="ignore"):
        for _ in range(steps):
            g = es.grad_centred(zn)
            pn[:, 0] -= 0.5 * eps * g[:, 0]
            pn[:, 2:] -= 0.5 * eps * g[:, 2:]
            pq = np.concatenate([pn[:, :1], pn[:, 2:]], 1)
            u = zn[:, 1]
            pn[:, 1] -= 0.5 * eps * (g[:, 1] + dlogdet_du(u) + dkin_du(u, pq))
            u_new = u + eps * pn[:, 1] / G_U
            ginv = 0.5 * (1.0 / masses(u) + 1.0 / masses(u_new))
            zn[:, 0] += eps * ginv[:, 0] * pn[:, 0]
            zn[:, 2:] += eps * ginv[:, 1:] * pn[:, 2:]
            zn[:, 1] = u_new
            g = es.grad_centred(zn)
            pn[:, 1] -= 0.5 * eps * (g[:, 1] + dlogdet_du(u_new) + dkin_du(u_new, pq))
            pn[:, 0] -= 0.5 * eps * g[:, 0]
            pn[:, 2:] -= 0.5 * eps * g[:, 2:]
        dh = hamiltonian(zn, pn) - h0
    div = ~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000)
    acc = ~div & (np.log(es.RNG.random(r)) < -np.nan_to_num(dh, nan=np.inf))
    return np.where(acc[:, None], zn, z), acc, div


def transition(sampler, kind, z, eps, steps):
    if sampler == "rmhmc":
        return rmhmc(z, eps, steps)
    es.STEPS = steps
    return es.hmc(kind, z, eps)


# ---------------------------------------------------------------- the pilots

def sbc_pilot(sampler, kind, truth, y, eps, steps, thins, pilot):
    """Burn, then pilot transitions with the ranks of the truth accumulated at each thin.

    Returns ranks per thin (r, 4), the divergence count per chain, the mean
    acceptance and the log tau pilot of IACT_CHAINS chains at thin 1.
    """
    es.Y = y
    r = y.shape[0]
    z = es.prior_start(kind, r)
    for _ in range(BURN):
        z, _, _ = transition(sampler, kind, z, eps, steps)
    below = {thin: np.zeros((r, 4), dtype=np.int32) for thin in thins}
    div = np.zeros(r)
    acc = 0.0
    kept = np.empty((pilot, IACT_CHAINS))
    for k in range(1, pilot + 1):
        z, a, d = transition(sampler, kind, z, eps, steps)
        div += d
        acc += a.mean()
        kept[k - 1] = z[:IACT_CHAINS, 1]
        p = params(kind, z)
        for thin in thins:
            if k % thin == 0 and k // thin <= L:
                below[thin] += p < truth
    return below, div, acc / pilot, kept


def ends(ranks):
    """Share of ranks in the two end tenths, 0.2 under uniform ranks."""
    g = ranks * 10 // (L + 1)
    return np.mean((g == 0) | (g == 9))


def print_info(label, acc, div, kept):
    med, q90, bad, rho1 = iact_summary(kept)
    print(f"  {label:30s} acc {acc:.3f} div {100 * div.mean() / kept.shape[0]:6.3f}% of proposals, "
          f"{100 * (div > 0).mean():5.1f}% of chains  iact median {med:5.1f} 90% {q90:6.1f}  "
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

    truth, y = draw_studies(r)
    print(f"eight schools, step size and the neck: {studies} x {nmax} fresh studies, burn {BURN}, L = {L}")
    print(f"  the prior studies: tau < {es.NECK:.0f} on {100 * np.mean(np.exp(truth[:, 1]) < es.NECK):.1f}%, "
          f"median tau {np.median(np.exp(truth[:, 1])):.2f}")

    runs = {}                    # label -> (ranks per thin (studies, nmax, 4), div per chain, steps)
    print("\n1. the pilots: acceptance, divergences, IACT of log tau from the pilot, nan guard, lag-1")
    for kind, sampler, eps, steps in TRAJECTORY:
        below, div, acc, kept = sbc_pilot(sampler, kind, truth, y, eps, steps, (1,), L)
        label = f"{kind} hmc eps {eps} x {steps} steps"
        runs[label] = ({1: below[1].reshape(studies, nmax, 4)}, div, steps)
        print_info(label, acc, div, kept)
    for eps in CENTRED_EPS:
        below, div, acc, kept = sbc_pilot("hmc", "centred", truth, y, eps, 20, (1, 5), PILOT)
        label = f"centred hmc eps {eps} x 20 steps"
        runs[label] = ({t: below[t].reshape(studies, nmax, 4) for t in (1, 5)}, div, 20)
        print_info(label, acc, div, kept)
    for eps in RMHMC_EPS:
        thins, pilot = ((1, 5), PILOT) if eps == RMHMC_THIN_EPS else ((1,), L)
        below, div, acc, kept = sbc_pilot("rmhmc", "centred", truth, y, eps, 20, thins, pilot)
        label = f"centred rmhmc eps {eps} x 20 steps"
        runs[label] = ({t: below[t].reshape(studies, nmax, 4) for t in thins}, div, 20)
        print_info(label, acc, div, kept)
    print(f"  {time.time() - t0:.0f} s", flush=True)

    print(f"\n2. share of {studies} studies rejected, any of the four parameters, " + " / ".join(KEYS))
    for label, (ranks, _, _) in runs.items():
        for thin, rr in ranks.items():
            cells = "  ".join(f"n={n}: " + "/".join(f"{d2.reject_any(rr[:, :n], stats[n], crits[n])[k]:.2f}" for k in KEYS)
                              for n in SIZES)
            print(f"  {label:30s} thin {thin:2d} {cells}")

    print(f"\n3. each parameter alone at n = 100 and {nmax}, band / mean, " + " / ".join(PARAMS))
    for label, (ranks, _, _) in runs.items():
        for thin, rr in ranks.items():
            cells = []
            for n in (100, nmax):
                b = per_param(rr[:, :n], stats[n], crits[n], "band")
                m = per_param(rr[:, :n], stats[n], crits[n], "mean")
                cells.append(f"n={n}: " + " ".join(f"{bb:.2f}/{mm:.2f}" for bb, mm in zip(b, m)))
            print(f"  {label:30s} thin {thin:2d} " + "  ".join(cells))

    print(f"\n4. pooled rank histograms in tenths at n = {nmax}")
    for label, (ranks, _, _) in runs.items():
        for thin, rr in ranks.items():
            print(f"  {label:30s} thin {thin:2d} mu      {shape(rr, 0)}")
            for p, name in ((1, "log tau"), (2, "theta_1"), (3, "eta_1  ")):
                print(f"  {'':30s} {'':7s} {name} {shape(rr, p)}")

    print(f"\n5. the cost of a passing study: band on the four together at n = {nmax}, gradients a simulation = thin x steps x {L}")
    print("  a pass is a rejection share at or under 0.25, the nominal 0.19 of four tests at 5% plus the error of 40 studies")
    for label, (ranks, div, steps) in runs.items():
        for thin, rr in ranks.items():
            share = d2.reject_any(rr, stats[nmax], crits[nmax])["band"]
            verdict = "pass" if share <= 0.25 else "fail"
            print(f"  {label:30s} thin {thin:2d}  band {share:.2f}  {verdict}  {thin * steps * L:6d} gradients  "
                  f"div {100 * div.mean() / (PILOT if len(ranks) > 1 else L):6.3f}%")

    print("\n6. centred, hmc against rmhmc: share of each parameter's ranks in the end tenths at thin 1, 0.20 if uniform")
    for label, (ranks, _, _) in runs.items():
        if "centred" not in label or "noncentred" in label:
            continue
        rr = ranks[1].reshape(r, 4)
        print(f"  {label:30s} " + "  ".join(f"{name} {ends(rr[:, p]):.2f}" for p, name in enumerate(PARAMS)))

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
