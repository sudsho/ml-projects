"""
Day 3 of MCMC.

Day 2 ended with two separate costs: dimension, which is linear in `d`, and
conditioning, which is worse than linear and which one fix removes exactly. HMC
is supposed to improve the first. Today is about whether it does, what it does
to the second, and which of its advertised properties survive being written as
numbers.

Six things get measured.

  1. **The two identities the correctness argument actually uses.** HMC's
     Metropolis step is legitimate because the leapfrog map is volume preserving
     and reversible. Both are exact statements, not approximations, and neither
     needs the step size to be small - so both can be written as residuals. The
     Jacobian determinant of the `2d x 2d` map comes back 1.000000000091 at
     `eps = 0.1` and 0.999999992267 at `eps = 2.0`, where the energy error is
     already 51.1 and every proposal is being rejected. Reversibility residuals
     over the same range are 1e-16 to 1e-14. That is the useful part: the
     integrator being wrong about the physics does not make the sampler wrong
     about the distribution.

     At `eps = 5.0` the determinant reads `-9.0e17` and the reversibility
     residual reads 6.4e8. Neither identity has stopped holding. The trajectory
     has reached magnitude 1e9, so the finite-difference perturbation I use to
     build the Jacobian is amplified past the point where the difference means
     anything, and the same amplification pushes the round-trip's rounding error
     up with it. Both checks fail together, at the same step size, for a reason
     that has nothing to do with either property. I am measuring my instrument
     there, and it is worth knowing where that starts.

  2. **The energy error, and an identity whose estimator is worse than it
     looks.** Leapfrog is second order, so `|dH|` at fixed trajectory length
     should fall like `eps^2`. Measured ratios as `eps` halves: 3.47, 4.02, 4.01,
     4.00. That is as clean as anything in this project has been.

     The other check is `E[exp(-dH)] = 1`, an exact identity over proposals that
     follows from volume preservation, and much sharper than watching `|dH|` be
     small. It reads 1.0000 at every stable step size. Then it reads 0.9398 at
     `eps = 1.8` and 0.0000 at `eps = 2.1`, where nothing about the identity has
     changed. `exp(-dH)` is dominated by rare proposals with large negative
     `dH`, and the mean of 4000 draws does not contain them, so the estimator is
     biased low by an amount that grows exactly as the integrator degrades. The
     sharpest available check is unreliable in precisely the regime it exists to
     detect, and it fails toward "looks fine", not toward a warning.

  3. **The dimension scaling.** Cost per effectively independent sample, counted
     in target evaluations: HMC 8.0, 8.0, 14.1, 23.0, 38.5, 58.0 for `d` = 1, 2,
     5, 10, 25, 50; RWM 4.4, 7.9, 17.7, 32.8, 84.1, 179.2. So HMC grows 7.2x
     while `d` grows 50x and RWM grows 41x, and it is worth 3.1x at `d = 50`.

     Two caveats and they matter more than the number. HMC *loses* at `d <= 2`,
     because its floor is `L` gradients per sample no matter how good the
     trajectory is, and RWM's floor is 1. The crossover is between `d = 2` and
     `d = 5`. And the acceptance rate is 0.995 at every dimension, which day 1
     and part 5 both say is far above the efficient band, so this is a
     conservatively tuned HMC against a tuned RWM. 3.1x is a lower bound on the
     comparison and not a measurement of the method.

  4. **Conditioning, where the gradient does not help.** At `d = 2` with an
     identity mass matrix: IACT 1.00, 21.8, 116.6, 334.9 for kappa = 1, 19, 100,
     400. Day 2's RWM on the same targets: 8.0, 61.8, 310.6, 901.7. The ratio is
     roughly 2.7x and roughly constant, which is the point - HMC pays the same
     growth in kappa that RWM pays, at a smaller constant. Knowing the gradient
     does not tell you the scale of the direction you are moving in.

     Setting the mass matrix to `Sigma^-1` gives IACT 1.00 at kappa 19, 100 and
     400 alike. That is day 2's affine-invariance result again, unchanged: the
     preconditioned chain on the correlated target is the isotropic chain,
     relabelled. HMC's version looks better only because HMC's isotropic number
     is 1.00 where RWM's was 8.0. The fix is the same fix, it is still the thing
     that matters more than the sampler, and day 2's finding that a rough pilot
     covariance is enough should carry over unchanged.

  5. **Trajectory length, and the day's actual finding.** On `N(0, I_10)` at
     `eps = 0.2`, sweeping the number of leapfrog steps: IACT 1.02 at `L = 8`,
     1.00 at `L = 16`, then **988.9 at `L = 31`**, 340.4 at 32, 57.2 at 33, and
     1.00 again at 48. The free Hamiltonian trajectory on a Gaussian is a
     rotation of period `2 pi`, and `31 * 0.2 = 6.20` against `2 pi = 6.283`. The
     sampler is integrating almost exactly one full orbit and proposing the point
     it started from. `L = 64` gives `L eps = 12.8` against `4 pi = 12.57` and
     IACT 93.7, the same resonance one harmonic out and weaker.

     Three percent in `L` is a factor of a thousand in cost, and there is no
     signal in the acceptance rate: `L = 31` accepts 0.999, the highest in the
     entire sweep, while producing a chain a thousand times worse than its
     neighbours. Day 1 got this moral from a broken sampler whose mean was right,
     day 2 got it from an independence sampler that ranked backwards by
     acceptance, and this is the third and cleanest instance. Acceptance rate is
     a property of the proposal.

     Jittering `L` by +/-20% takes 988.9 to 7.4, 340.4 to 8.2, 57.2 to 5.2. It
     removes the catastrophe and does not restore the good case, since jittered
     `L = 16` still runs at 1.00. Jitter is insurance with a premium, not a free
     improvement, and I had been carrying it as a free improvement.

     The step size sweep, by contrast, has no interior optimum to find. At fixed
     `L eps ~ 2` the IACT is pinned at the estimator's floor of 1.00 for every
     `eps` up to 0.6, so cost is just `L` and falls monotonically: 40.0, 20.0,
     10.0, 7.0, 5.0, 3.0, 2.9 gradients per sample. The rule of thumb points at
     acceptance 0.65 and the cheapest points measured accept 0.78 and 0.88. Same
     shape as day 1's 0.234: the target is roughly right and the sampler is
     insensitive to it over the range where it matters.

  6. **Where it is meant to struggle, and a mistake I made reading it.** On the
     banana, HMC at `eps = 0.15, L = 20` runs at IACT 82.7 against 252.4 for
     day 2's best RWM, with relative covariance error 0.247 against 0.073. I
     wrote that down as HMC mixing three times faster and being three times
     wronger. It is not: 8000 HMC steps at IACT 83 hold 87 effective samples and
     120000 RWM steps at IACT 252 hold 428, so the accuracy comparison was
     between runs differing 5x in information. Matched on ESS instead - 40000
     HMC steps, ESS 584 - the error is 0.028 against 0.073. HMC is better on both
     axes and the original reading was an artefact of comparing at equal chain
     length, which is the wrong axis whenever the IACTs differ.

     Then Neal's funnel at `d = 3`, where `Var(v) = 9` and `Var(x_i) = e^4.5 =
     90.02` exactly. Three step sizes give `Var(v)` = 7.26, 7.11, 7.32, all wrong
     by the same 20% and indistinguishable from each other, so the moment that is
     easiest to check does not separate them at all. `Var(x_0)` gives 9.56,
     18.43, 70.45 against 90.02. The divergence counts are 7, 142, 841, ranking
     the three runs in exactly the reverse order of their `Var(x_0)` accuracy.

     What I will not conclude is that the divergent sampler is better. `min(v)`
     is -7.29, -5.52, -3.66: the small step size climbs deep into the neck and
     stays there, the large one never enters it, and neither covers the funnel.
     They are differently wrong, and one of the two ways of being wrong happens
     to land nearer the true variance of `x`. That the diagnostic sold for
     detecting this ranks them backwards is day 4's problem, which is what day 4
     is for.

Run: `python day3_hmc.py`
"""

import numpy as np

from day1_metropolis import (
    CorrelatedGaussian,
    IIDGaussian,
    integrated_act,
    random_walk_metropolis,
)
from day2_mixing import Banana, anisotropic_cov, preconditioned_metropolis


# --- gradients ---------------------------------------------------------------


def grad_log_density(target, x):
    """Analytic gradient of `log pi` for the targets this project uses.

    HMC is the first sampler here that needs more than a density evaluation, and
    the gradient is the whole reason it works, so it gets written out per target
    rather than differenced. `fd_grad` below exists to check these, once, and is
    not used in any chain.
    """
    if isinstance(target, IIDGaussian):
        return -x
    if isinstance(target, CorrelatedGaussian):
        return -target.prec @ (x - target.mu)
    if isinstance(target, Banana):
        b = target.b
        r = x[1] - b * (x[0] ** 2 - 100.0)
        return np.array([-x[0] / 100.0 + r * 2.0 * b * x[0], -r])
    # targets that carry their own analytic gradient answer for themselves. the
    # branch was `isinstance(target, Funnel)` until day 4 added a second one;
    # duck-typing it is the smaller change and keeps the closed list above for
    # the targets that were written before there was a gradient to write.
    if hasattr(target, "grad_log_density"):
        return target.grad_log_density(x)
    raise TypeError(f"no analytic gradient for {type(target).__name__}")


def fd_grad(target, x, h=1e-5):
    """Central-difference gradient. Only ever used to check the analytic ones."""
    g = np.empty_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = h
        g[k] = (target.log_density(x + e) - target.log_density(x - e)) / (2 * h)
    return g


class Funnel:
    """Neal's funnel: `v ~ N(0, 3^2)`, `x_i | v ~ N(0, exp(v))` for `i < d-1`.

    Here early because day 4 needs it, and because it is the one target where a
    step size that works at one point in the state space cannot work at another -
    the conditional scale of `x` spans `exp(+/-3)`, a factor of 400 in each
    direction. Moments are closed form: `E[v] = 0`, `Var(v) = 9`, `E[x_i] = 0`,
    `Var(x_i) = E[exp(v)] = exp(9/2)`.
    """

    def __init__(self, d=3, sigma_v=3.0):
        self.d, self.sigma_v, self.name = d, sigma_v, f"funnel-d{d}"

    def log_density(self, z):
        v, x = z[-1], z[:-1]
        return (
            -0.5 * (v / self.sigma_v) ** 2
            - 0.5 * np.dot(x, x) * np.exp(-v)
            - 0.5 * (self.d - 1) * v
        )

    def grad_log_density(self, z):
        v, x = z[-1], z[:-1]
        g = np.empty(self.d)
        g[:-1] = -x * np.exp(-v)
        g[-1] = -v / self.sigma_v**2 + 0.5 * np.dot(x, x) * np.exp(-v) - 0.5 * (self.d - 1)
        return g

    def mean(self):
        return np.zeros(self.d)

    def cov(self):
        c = np.eye(self.d) * np.exp(0.5 * self.sigma_v**2)
        c[-1, -1] = self.sigma_v**2
        return c


# --- leapfrog ----------------------------------------------------------------


def leapfrog(target, q, p, eps, n_leap, inv_mass=None):
    """`n_leap` steps of the leapfrog integrator on `H(q,p) = -log pi(q) + K(p)`.

    Half-step the momentum, full-step the position, half-step the momentum, and
    the interior half-steps of consecutive iterations merge into one full step -
    which is why `n_leap` steps cost `n_leap` gradient evaluations and not
    `2 n_leap`.

    Two properties matter and both are checkable rather than assumed. The map is
    volume preserving, because each of the three sub-steps is a shear (one
    variable updated by a function of the other only), so each has unit Jacobian
    determinant. And it is reversible: negate `p` at the end, integrate again,
    and you land exactly where you started. `leapfrog_jacobian` and
    `reversibility_residual` measure both. Neither depends on the step size being
    small - they hold for a wildly unstable integrator too, which is the point.
    `H` itself is *not* conserved, and the amount by which it is not is what the
    accept step is for.
    """
    q = np.array(q, float)
    p = np.array(p, float)
    g = grad_log_density(target, q)

    p = p + 0.5 * eps * g
    for i in range(n_leap):
        q = q + eps * (inv_mass @ p if inv_mass is not None else p)
        g = grad_log_density(target, q)
        p = p + (eps if i < n_leap - 1 else 0.5 * eps) * g
    return q, p


def hamiltonian(target, q, p, inv_mass=None):
    k = 0.5 * (p @ inv_mass @ p if inv_mass is not None else p @ p)
    return -target.log_density(q) + k


def leapfrog_jacobian(target, q, p, eps, n_leap, h=1e-6):
    """Jacobian determinant of the leapfrog map, by finite differences.

    The claim is `det = 1` exactly, for any step size, stable or not. Measuring
    it costs `4 d` trajectories and is worth it once.
    """
    d = len(q)
    z0 = np.concatenate([q, p])

    def phi(z):
        a, b = leapfrog(target, z[:d], z[d:], eps, n_leap)
        return np.concatenate([a, b])

    J = np.empty((2 * d, 2 * d))
    for k in range(2 * d):
        e = np.zeros(2 * d)
        e[k] = h
        J[:, k] = (phi(z0 + e) - phi(z0 - e)) / (2 * h)
    return float(np.linalg.det(J))


def reversibility_residual(target, q, p, eps, n_leap):
    """`|q - q''|` after integrating forward, negating `p`, and integrating again."""
    q1, p1 = leapfrog(target, q, p, eps, n_leap)
    q2, p2 = leapfrog(target, q1, -p1, eps, n_leap)
    return float(np.abs(q2 - q).max()), float(np.abs(-p2 - p).max())


# --- the sampler -------------------------------------------------------------


def hmc(target, q0, n_steps, eps, n_leap, rng, inv_mass=None, jitter=0.0):
    """Hamiltonian Monte Carlo. Returns chain, acceptance rate, energy errors.

    Draw a momentum from `N(0, M)`, integrate for `n_leap` leapfrog steps, accept
    with probability `min(1, exp(-Delta H))`. The proposal is deterministic given
    the momentum, and the Metropolis correction is legitimate because the map is
    volume preserving and (with the momentum flip, which is implicit here since
    `K` is even) reversible - the two identities checked above are exactly the
    two the correctness argument uses.

    `jitter` randomises the number of leapfrog steps uniformly in
    `[(1-jitter) L, (1+jitter) L]`. On a Gaussian the trajectory is a rotation
    with a fixed period, so a fixed `L` can land back near where it started every
    time; the jitter is the standard defence, and part 5 measures whether it is
    needed here.

    The returned `dH` array is every proposed energy error including rejected
    ones, because `E[exp(-Delta H)] = 1` is an identity over proposals and is the
    sharpest available check that the integrator and the accept step agree.
    """
    d = len(q0)
    q = np.array(q0, float)
    chain = np.empty((n_steps, d))
    dH = np.empty(n_steps)
    n_accept = 0
    n_grad = 0

    chol_mass = None
    if inv_mass is not None:
        chol_mass = np.linalg.cholesky(np.linalg.inv(inv_mass))

    for t in range(n_steps):
        z = rng.standard_normal(d)
        p = chol_mass @ z if chol_mass is not None else z
        L = n_leap
        if jitter > 0.0:
            lo = max(1, int(round((1 - jitter) * n_leap)))
            hi = max(lo, int(round((1 + jitter) * n_leap)))
            L = int(rng.integers(lo, hi + 1))

        h0 = hamiltonian(target, q, p, inv_mass)
        qn, pn = leapfrog(target, q, p, eps, L, inv_mass)
        n_grad += L
        h1 = hamiltonian(target, qn, pn, inv_mass)

        delta = h1 - h0
        dH[t] = delta
        if not np.isfinite(delta):
            delta = np.inf
        if np.log(rng.random()) < -delta:
            q = qn
            n_accept += 1
        chain[t] = q

    return chain, n_accept / n_steps, dH, n_grad


def worst_iact(chain, burn, k_max=5):
    d = chain.shape[1]
    return max(integrated_act(chain[burn:, k]) for k in range(min(d, k_max)))


# --- run ---------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("day 3 - hamiltonian monte carlo, and which of its properties are exact")
    print("=" * 78)

    rng0 = np.random.default_rng(11)

    # 0. the gradients, before anything depends on them
    print("\n[0] analytic gradients vs central differences")
    for tgt, x in (
        (IIDGaussian(3), rng0.standard_normal(3)),
        (CorrelatedGaussian([0.5, -1.0], anisotropic_cov(2, 19, np.random.default_rng(7))),
         rng0.standard_normal(2)),
        (Banana(0.03), np.array([4.0, -2.0])),
        (Funnel(3), np.array([0.7, -0.4, 1.1])),
    ):
        err = np.abs(grad_log_density(tgt, x) - fd_grad(tgt, x)).max()
        print(f"    {tgt.name:22s}  max |analytic - fd| = {err:.2e}")

    # 1. the two identities the correctness argument rests on
    print("\n[1] volume preservation and reversibility, at step sizes both sane and not")
    print("    det of the 2d x 2d leapfrog Jacobian should be 1 whatever eps is")
    tgt = CorrelatedGaussian([0.0, 0.0], anisotropic_cov(2, 19, np.random.default_rng(7)))
    q = np.array([0.3, -0.2])
    p = np.array([1.0, 0.5])
    for eps in (0.01, 0.1, 0.5, 2.0, 5.0):
        det = leapfrog_jacobian(tgt, q, p, eps, 10)
        rq, rp = reversibility_residual(tgt, q, p, eps, 10)
        h0 = hamiltonian(tgt, q, p)
        qn, pn = leapfrog(tgt, q, p, eps, 10)
        dh = hamiltonian(tgt, qn, pn) - h0
        print(
            f"    eps={eps:5.2f}  det J = {det:.12f}   |dq|={rq:.2e}  |dp|={rp:.2e}"
            f"   dH={dh:12.4f}"
        )

    # 2. energy error scaling, and the identity E[exp(-dH)] = 1
    print("\n[2] energy error vs step size (L eps held at 1.0, so same trajectory length)")
    print("    leapfrog is 2nd order: global |dH| should fall like eps^2")
    tgt = IIDGaussian(10)
    prev = None
    for eps in (0.4, 0.2, 0.1, 0.05, 0.025):
        L = int(round(1.0 / eps))
        rng = np.random.default_rng(5)
        _, acc, dH, _ = hmc(tgt, np.zeros(10), 4000, eps, L, rng)
        m = float(np.mean(np.abs(dH)))
        ratio = "" if prev is None else f"  ratio={prev/m:5.2f}"
        prev = m
        print(
            f"    eps={eps:6.3f}  L={L:3d}  acc={acc:.3f}  mean|dH|={m:.3e}"
            f"  E[exp(-dH)]={float(np.mean(np.exp(-dH))):.4f}{ratio}"
        )

    print("\n    the same identity on a deliberately unstable integrator")
    for eps in (1.0, 1.8, 2.1):
        rng = np.random.default_rng(5)
        _, acc, dH, _ = hmc(tgt, np.zeros(10), 4000, eps, 10, rng)
        fin = np.isfinite(dH)
        print(
            f"    eps={eps:6.3f}  L= 10  acc={acc:.3f}  mean|dH|={np.mean(np.abs(dH[fin])):.3e}"
            f"  E[exp(-dH)]={float(np.mean(np.exp(-dH[fin]))):.4f}"
            f"  finite {fin.sum()}/{len(dH)}"
        )

    # 3. the dimension scaling day 2 ended on
    print("\n[3] dimension scaling: RWM was linear in d. HMC, at matched gradient cost")
    print("    HMC eps=0.25/d^0.25, L=8; cost counted in density/gradient evaluations")
    n_hmc = 8000
    n_rwm = 120_000
    for d in (1, 2, 5, 10, 25, 50):
        tgt = IIDGaussian(d)
        eps = 0.25 / d**0.25
        L = 8
        rng = np.random.default_rng(4242 + d)
        ch, acc, dH, ng = hmc(tgt, np.zeros(d), n_hmc, eps, L, rng)
        iact_h = worst_iact(ch, n_hmc // 10)

        rng = np.random.default_rng(4242 + d)
        chr_, accr = random_walk_metropolis(tgt, np.zeros(d), n_rwm, 2.3 / np.sqrt(d), rng)
        iact_r = worst_iact(chr_, n_rwm // 10)

        # cost per effectively independent sample, in target evaluations
        cost_h = iact_h * L
        print(
            f"    d={d:3d}  HMC acc={acc:.3f} IACT={iact_h:7.2f} grads/ESS={cost_h:8.1f}"
            f"   |  RWM acc={accr:.3f} IACT={iact_r:8.1f} evals/ESS={iact_r:8.1f}"
        )

    # 4. conditioning, which is what actually cost day 2
    print("\n[4] conditioning at d=2: does the gradient fix what preconditioning fixed?")
    n_steps = 8000
    for kappa in (1, 19, 100, 400):
        sigma = (
            np.eye(2) if kappa == 1 else anisotropic_cov(2, kappa, np.random.default_rng(7))
        )
        tgt = CorrelatedGaussian([0.0, 0.0], sigma)
        lam_min = np.linalg.eigvalsh(sigma).min()
        eps = 0.25 * np.sqrt(lam_min)
        rng = np.random.default_rng(99)
        ch, acc, _, _ = hmc(tgt, np.zeros(2), n_steps, eps, 8, rng)
        iact = worst_iact(ch, n_steps // 10)

        # and with a mass matrix set to the target precision, the HMC analogue
        # of day 2's preconditioner
        rng = np.random.default_rng(99)
        ch2, acc2, _, _ = hmc(
            tgt, np.zeros(2), n_steps, 0.25, 8, rng, inv_mass=sigma
        )
        iact2 = worst_iact(ch2, n_steps // 10)
        print(
            f"    kappa={kappa:4d}  identity mass acc={acc:.3f} IACT={iact:8.2f}"
            f"   |  M=Sigma^-1 acc={acc2:.3f} IACT={iact2:7.2f}"
        )

    # 5. tuning: the acceptance optimum, and the trajectory-length trap
    print("\n[5] tuning. step size first: sweep eps at fixed trajectory length L*eps~2")
    tgt = IIDGaussian(10)
    for eps in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8):
        L = max(1, int(round(2.0 / eps)))
        rng = np.random.default_rng(77)
        ch, acc, _, ng = hmc(tgt, np.zeros(10), 6000, eps, L, rng)
        iact = worst_iact(ch, 600)
        print(
            f"    eps={eps:5.2f}  L={L:3d}  acc={acc:.3f}  IACT={iact:7.2f}"
            f"  grads per ESS={iact*L:9.1f}"
        )

    print("\n    trajectory length at fixed eps=0.2 on N(0, I_10)")
    print("    the free trajectory is a rotation of period 2pi, so L*eps near 2pi")
    print("    should come back to where it started")
    for L in (2, 4, 8, 16, 24, 31, 32, 33, 48, 64):
        rng = np.random.default_rng(77)
        ch, acc, _, _ = hmc(tgt, np.zeros(10), 6000, 0.2, L, rng)
        iact = worst_iact(ch, 600)
        print(
            f"    L={L:3d}  L*eps={0.2*L:5.2f}  acc={acc:.3f}  IACT={iact:7.2f}"
            f"  grads per ESS={iact*L:9.1f}"
        )

    print("\n    and the same sweep with the number of steps jittered +/-20%")
    for L in (16, 31, 32, 33):
        rng = np.random.default_rng(77)
        ch, acc, _, _ = hmc(tgt, np.zeros(10), 6000, 0.2, L, rng, jitter=0.2)
        iact = worst_iact(ch, 600)
        print(f"    L={L:3d} jittered  acc={acc:.3f}  IACT={iact:7.2f}")

    # 6. where it is supposed to fail
    print("\n[6] the targets with no single good step size")
    ban = Banana(0.03)
    print("    covariance error is compared at matched ESS, not matched chain length -")
    print("    otherwise the cheaper-per-sample sampler is credited for a longer run")
    rng = np.random.default_rng(5)
    chp, accp = preconditioned_metropolis(
        ban, np.zeros(2), 120_000, np.linalg.cholesky(ban.cov()) * (2.3 / np.sqrt(2)), rng
    )
    iact_p = worst_iact(chp, 12_000)
    ess_p = (120_000 - 12_000) / iact_p
    err_p = np.abs(np.cov(chp[12_000:].T) - ban.cov()).max() / np.abs(ban.cov()).max()
    print(
        f"    banana  day-2 precond RWM     acc={accp:.3f}  IACT={iact_p:7.2f}"
        f"  ESS={ess_p:7.1f}  rel cov err={err_p:.3f}"
    )
    for n_steps in (8_000, 40_000):
        rng = np.random.default_rng(5)
        ch, acc, _, _ = hmc(ban, np.zeros(2), n_steps, 0.15, 20, rng)
        b = n_steps // 10
        iact = worst_iact(ch, b)
        err = np.abs(np.cov(ch[b:].T) - ban.cov()).max() / np.abs(ban.cov()).max()
        print(
            f"    banana  HMC eps=0.15 L=20  acc={acc:.3f}  IACT={iact:7.2f}"
            f"  ESS={(n_steps-b)/iact:7.1f}  rel cov err={err:.3f}  ({n_steps} steps)"
        )

    fun = Funnel(3)
    print("\n    Neal's funnel, d=3: Var(v)=9 and Var(x_i)=exp(4.5)=90.02 exactly")
    for eps in (0.05, 0.15, 0.4):
        rng = np.random.default_rng(3)
        ch, acc, dH, _ = hmc(fun, np.zeros(3), 20_000, eps, 20, rng)
        b = 2000
        vv = ch[b:, -1]
        print(
            f"    eps={eps:5.2f}  acc={acc:.3f}  IACT(v)={integrated_act(vv):7.2f}"
            f"  Var(v)={vv.var():6.2f}  Var(x0)={ch[b:,0].var():8.2f}"
            f"  min v={vv.min():6.2f}  divergences={int(np.sum(dH > 1000))}"
        )

    print("\ndone.")
