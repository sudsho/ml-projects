"""
Day 4 of MCMC, and the last one.

Days 1 to 3 all ended in the same place from three directions: a number that is
supposed to tell you how the sampler is doing does not. Day 1's ESS estimator
called a fully stuck chain the most efficient point on the grid. Day 2's
uniform-ergodicity threshold was binary where the performance was continuous.
Day 3's acceptance rate was 0.999 at the worst trajectory length in the sweep,
the highest in the whole run. Today is the target where that failure is the
point rather than an aside, and the diagnostics get measured instead of the
sampler.

Neal's funnel: `v ~ N(0, 3^2)`, `x_i | v ~ N(0, exp(v))`. The conditional scale
of `x` is `exp(v/2)`, so it spans a factor of 400 in each direction over the
range `v` actually visits, and no single step size can be right everywhere. The
moments are closed form - `Var(v) = 9`, `Var(x_i) = exp(4.5) = 90.02` - so
everything below is measured against a known answer.

Six things get measured.

  1. **The failure, and the moment that does not see it.** HMC on the centred
     funnel, 20k steps, `L = 20`, step size swept. `Var(v)` comes back 8.49,
     8.01, 8.30, 9.45, 6.54 for `eps` = 0.05 to 0.8 against a true 9.00. Four of
     those five are within 8%. If `Var(v)` were the check, the sampler passes.

     The chain is nowhere near right. `min v` over the same sweep is -7.48,
     -6.01, -5.07, -3.97, -2.71, and `v` has standard deviation 3, so the best
     of those reaches 2.5 sigma and the worst reaches 0.9. The neck of the funnel
     is not being visited at all. A marginal variance is an average over the bulk
     and the bulk is fine; what is missing is a tail that contributes little to
     the second moment and is the entire difficulty of the target.

     The IACT tells the same story more loudly - 99.3, 47.3, 47.0, 1814.4, 747.2
     - and it is not monotone in `eps`, so it cannot be read as a tuning curve
     either. And the acceptance rate is 0.984 at `eps = 0.05`, the highest in the
     sweep and the second worst IACT in it. That is the fourth time in four days.

  2. **The reparameterisation, which is not a better sampler.** Write
     `x_i = xt_i * exp(v/2)` with `xt ~ N(0, I)` and the target becomes a
     product of independent Gaussians - the funnel moves out of the distribution
     and into a transform applied to the output afterwards. Same HMC, same step
     sizes, same seeds: `Var(v)` = 9.58, 8.98, 8.98, 8.94, 8.96, `min v` = -10.6,
     -12.4, -11.5, -11.1, -13.0, and **zero divergences at every step size**.
     IACT drops to 1.63 at `eps = 0.2` and 1.00 at `eps = 0.4`, which is
     independent draws.

     `min v` is the number worth keeping. It moves from -6.0 to -12.4 at the same
     step size on the same sampler, and -12.4 is past 4 sigma. Nothing about the
     sampler changed; a person supplied the coordinates.

  3. **`Var(x)` is a bad estimand even when the sampler is right.** True value
     90.02. Centred: 18.3, 33.5, 78.1, 132.9, 75.8, wrong and non-monotone.
     Non-centred: 69.9, 70.3, 61.8, 61.1, 135.4, also wrong, and that one is not
     the sampler's fault. `E[x^2] = E[exp(v)]` is dominated by the rare large-`v`
     draws that 18k samples do not contain, so the estimator is biased low for
     the same reason day 3's `E[exp(-dH)] = 1` check was: a heavy-tailed
     expectation estimated by a mean. Two of the four days have now produced a
     diagnostic that fails because of its estimator rather than its definition.

  4. **R-hat, and what it is a test of.** Four dispersed chains, 8k steps each.
     Centred at `eps = 0.10`: R-hat 1.0041, split R-hat 1.0045. Both pass the
     conventional 1.01 threshold comfortably. The same chains have 42
     divergences, `Var(v) = 7.39` against 9.00, and never get below `v = -6.14`.
     R-hat certifies it. At `eps = 0.40` R-hat is 1.3573 and does fire, but that
     is the case where 3618 divergences and an IACT of 1814 had already said so.

     Three controls make the reading precise:

       - four chains frozen at four different points: R-hat `inf`, caught - but
         caught because `W = 0` and the statistic divides by it, not because it
         returned a large number;
       - four chains each sampling `N(5, 1)` when the target is `N(0, 1)`:
         R-hat 1.0000, not caught;
       - four chains frozen at the *same* point: R-hat 0.9999, and that value is
         `sqrt((n-1)/n) = 0.9998750` to seven places. `W` is 1.2e-32 instead of 0
         only because the two-pass variance leaves rounding behind, so the
         statistic is 0/0 and what is printed is a function of the chain length
         and of nothing else.

     R-hat is a test of whether the chains agree with each other. Agreement and
     correctness are different properties and the middle control is four chains
     that agree perfectly about the wrong distribution. This is day 1's ESS
     result again in a different statistic: a degenerate chain gets the best
     score the diagnostic can award.

  5. **What the split buys, measured.** Four chains from a common far start, all
     relaxing the same way: unsplit R-hat 0.9999, split R-hat 1.7187. The drift
     is identical across chains, so it lives entirely inside `W` where it reads
     as variance, and the between-chain term has nothing to compare. Cutting
     each chain in half puts the early and late halves in competition and the
     drift becomes a difference of means. On genuinely stationary chains the two
     agree, 0.9999 and 1.0000.

     Note also that the unsplit statistic comes back *below* 1 on a chain that
     has not converged. There is no floor at 1; the `(n-1)/n` factor drags it
     under whenever the between-chain term is negligible, which is exactly the
     unconverged-in-parallel case.

  6. **The divergence count is the one that tracks the difficulty.** Fixing the
     sampler and sweeping `sigma_v`, which is what sets how deep the neck goes:
     divergences 0, 21, 113, 4204 for `sigma_v` = 1, 2, 3, 4, monotone and
     spanning three orders of magnitude. The variance ratio over the same sweep
     is 1.017, 0.948, 0.728, 0.765 - wrong, but not monotone, so it cannot even
     be used to rank the four problems by how hard they are.

     That is the day's actual finding. The diagnostic that works here is the one
     that reports where the sampler *failed to go*, and it works because a
     divergence is a trajectory the integrator lost in the region being missed.
     Every diagnostic that failed above - the marginal variance, the acceptance
     rate, R-hat - is computed from the draws that were collected, and a region
     the chain never enters leaves no trace in them.

Run: `python day4_diagnostics.py` (about 50s).
"""

import time

import numpy as np

from day1_metropolis import integrated_act
from day3_hmc import Funnel, hmc


# --- the reparameterised funnel ----------------------------------------------


class NonCentredFunnel:
    """Neal's funnel in the coordinates that remove its geometry.

    The centred funnel has `x_i | v ~ N(0, exp(v))`, so the conditional scale of
    `x` is a function of the parameter being sampled and no single step size can
    be right everywhere. Write `x_i = xt_i * exp(v/2)` with `xt_i ~ N(0, 1)`
    independent of `v` and the dependence moves out of the distribution and into
    the transform:

        log pi(xt, v) = -0.5 (v/sigma_v)^2 - 0.5 ||xt||^2

    which is a product of independent Gaussians with condition number
    `sigma_v^2` and nothing else. The target the sampler sees has no funnel in
    it at all; the funnel is recovered afterwards by `to_centred`, which is a
    deterministic map applied to the output and costs nothing.

    Worth being explicit that this is not a better sampler. It is the same
    sampler on a different parameterisation, and the parameterisation is
    something a person supplied.
    """

    def __init__(self, d=3, sigma_v=3.0):
        self.d, self.sigma_v, self.name = d, sigma_v, f"funnel-nc-d{d}"

    def log_density(self, z):
        v, xt = z[-1], z[:-1]
        return -0.5 * (v / self.sigma_v) ** 2 - 0.5 * np.dot(xt, xt)

    def grad_log_density(self, z):
        g = np.empty(self.d)
        g[:-1] = -z[:-1]
        g[-1] = -z[-1] / self.sigma_v**2
        return g

    def to_centred(self, chain):
        """Map a chain in `(xt, v)` back to the funnel's own `(x, v)`."""
        out = np.array(chain, float)
        out[:, :-1] *= np.exp(0.5 * out[:, -1])[:, None]
        return out


# --- convergence diagnostics -------------------------------------------------


def rhat(chains):
    """Gelman-Rubin R-hat for `chains` of shape `(m, n)`, one scalar quantity.

    `W` is the mean within-chain variance, `B/n` the variance of the chain
    means, and the estimator compares a pooled variance estimate against `W`.
    The reading it supports is narrow and worth stating in full: R-hat near 1
    says the chains agree with each other about the variance of this quantity.
    It does not say they agree with the target, and nothing in the formula ever
    looks at the target.
    """
    chains = np.asarray(chains, float)
    m, n = chains.shape
    means = chains.mean(axis=1)
    w = chains.var(axis=1, ddof=1).mean()
    b_over_n = means.var(ddof=1)
    var_hat = (n - 1) / n * w + b_over_n
    return float(np.sqrt(var_hat / w))


def split_rhat(chains):
    """R-hat after cutting every chain in half.

    The split is the entire difference and it buys one specific thing: a single
    chain that drifts monotonically has two halves with different means, so the
    between-chain term picks it up. Unsplit R-hat over chains that each drift
    the same way cannot see it, because the drift is inside `W` where it looks
    like variance.
    """
    chains = np.asarray(chains, float)
    m, n = chains.shape
    half = n // 2
    return rhat(np.concatenate([chains[:, :half], chains[:, half : 2 * half]]))


def divergences(dH, threshold=1000.0):
    """Count proposals whose energy error exploded.

    A divergence is not a rejected proposal; it is a trajectory the integrator
    lost, and it is informative precisely because the region that produces it -
    the neck of the funnel, where the step size is far too large for the local
    scale - is the region the chain is failing to visit. The count is a report
    about where the sampler did not go.
    """
    return int(np.sum(~np.isfinite(dH)) + np.sum(np.nan_to_num(dH, nan=0.0) > threshold))


def run_chains(target, n_chains, n_steps, eps, n_leap, seed0, spread, transform=None):
    """`n_chains` HMC runs from overdispersed starts. Returns chains and dH."""
    chains, dhs, accs = [], [], []
    for k in range(n_chains):
        rng = np.random.default_rng(seed0 + k)
        q0 = rng.standard_normal(target.d) * spread
        ch, acc, dH, _ = hmc(target, q0, n_steps, eps, n_leap, rng)
        if transform is not None:
            ch = transform(ch)
        chains.append(ch)
        dhs.append(dH)
        accs.append(acc)
    return np.array(chains), np.array(dhs), float(np.mean(accs))


# --- the run -----------------------------------------------------------------


def main():
    t0 = time.time()
    d = 3
    fun, ncf = Funnel(d), NonCentredFunnel(d)
    true_var_v = fun.sigma_v**2
    true_var_x = np.exp(0.5 * fun.sigma_v**2)
    print("MCMC day 4 - the funnel, the reparameterisation, and the diagnostics")
    print(f"target: Neal's funnel d={d}.  Var(v)={true_var_v:.2f}  Var(x_i)={true_var_x:.2f}")

    print("\n[1] HMC on the centred funnel, step size swept")
    print("    the neck is at v << 0 where the conditional scale of x is exp(v/2)")
    for eps in (0.05, 0.1, 0.2, 0.4, 0.8):
        rng = np.random.default_rng(11)
        ch, acc, dH, _ = hmc(fun, np.zeros(d), 20_000, eps, 20, rng)
        v = ch[2000:, -1]
        print(
            f"    eps={eps:5.2f}  acc={acc:.3f}  div={divergences(dH):5d}"
            f"  Var(v)={v.var():6.2f}  min v={v.min():7.3f}"
            f"  IACT(v)={integrated_act(v):7.2f}  Var(x0)={ch[2000:, 0].var():9.2f}"
        )

    print("\n[2] the same sampler, same settings, non-centred coordinates")
    for eps in (0.05, 0.1, 0.2, 0.4, 0.8):
        rng = np.random.default_rng(11)
        ch, acc, dH, _ = hmc(ncf, np.zeros(d), 20_000, eps, 20, rng)
        cen = ncf.to_centred(ch)
        v = cen[2000:, -1]
        print(
            f"    eps={eps:5.2f}  acc={acc:.3f}  div={divergences(dH):5d}"
            f"  Var(v)={v.var():6.2f}  min v={v.min():7.3f}"
            f"  IACT(v)={integrated_act(v):7.2f}  Var(x0)={cen[2000:, 0].var():9.2f}"
        )

    print("\n[3] R-hat on four dispersed chains, both parameterisations")
    print("    the quantity is v, whose true variance is 9.00")
    for label, target, transform in (
        ("centred    ", fun, None),
        ("non-centred", ncf, ncf.to_centred),
    ):
        for eps in (0.1, 0.4):
            chs, dhs, acc = run_chains(
                target, 4, 8_000, eps, 20, seed0=200, spread=1.0, transform=transform
            )
            v = chs[:, 1000:, -1]
            div = sum(divergences(h) for h in dhs)
            print(
                f"    {label} eps={eps:4.2f}  R-hat={rhat(v):.4f}"
                f"  split R-hat={split_rhat(v):.4f}  div={div:5d}"
                f"  Var(v)={v.var():6.2f}  min v={v.min():7.3f}"
            )

    print("\n[4] what R-hat is actually testing")
    n4 = 4_000
    frozen = np.repeat(np.array([[-2.0], [-0.5], [0.5], [2.0]]), n4, axis=1)
    print(f"    frozen at four different points   R-hat={rhat(frozen):8.4f}   caught")
    agree = np.array(
        [5.0 + np.random.default_rng(700 + k).standard_normal(n4) for k in range(4)]
    )
    print(f"    four chains agreeing, all wrong    R-hat={rhat(agree):8.4f}   not caught")
    stuck = np.repeat(np.array([[0.3], [0.3], [0.3], [0.3]]), n4, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_stuck = rhat(stuck)
    print(f"    frozen at the SAME point          R-hat={r_stuck:8.4f}   not caught")
    print(f"      that last value is sqrt((n-1)/n) = {np.sqrt((n4 - 1) / n4):.7f} to 7 places.")
    print("      W is 1.2e-32 rather than 0 only because the two-pass variance leaves")
    print("      rounding behind, so the statistic is 0/0 and what gets printed is a")
    print("      function of the chain length alone.")

    print("\n[5] split R-hat against a drift the unsplit statistic cannot see")
    print("    four chains from a common far start, all relaxing the same way")
    n = 6_000
    drift = np.array(
        [
            20.0 * np.exp(-np.arange(n) / 2500.0)
            + np.random.default_rng(400 + k).standard_normal(n)
            for k in range(4)
        ]
    )
    print(f"    drifting          R-hat={rhat(drift):.4f}  split R-hat={split_rhat(drift):.4f}")
    settled = np.array(
        [np.random.default_rng(500 + k).standard_normal(n) for k in range(4)]
    )
    print(f"    stationary        R-hat={rhat(settled):.4f}  split R-hat={split_rhat(settled):.4f}")

    print("\n[6] the divergence rate as a function of how deep the neck is")
    print("    sigma_v sets the depth; the sampler and step size are held fixed")
    for sig in (1.0, 2.0, 3.0, 4.0):
        f = Funnel(d, sigma_v=sig)
        rng = np.random.default_rng(31)
        ch, acc, dH, _ = hmc(f, np.zeros(d), 12_000, 0.2, 20, rng)
        v = ch[1500:, -1]
        print(
            f"    sigma_v={sig:4.1f}  true Var(v)={sig**2:6.2f}  measured={v.var():6.2f}"
            f"  ratio={v.var() / sig**2:5.3f}  div={divergences(dH):5d}  acc={acc:.3f}"
        )

    print(f"\ndone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
