"""
Day 2 of MCMC.

Day 1 ended on a number I did not go looking for: `N(0, I_2)` runs at IACT 8.1
and a 2d Gaussian with condition number 19 runs at 46.8, worse than `d = 50`
isotropic. Two parameters beat fifty because they were correlated. That is what
today is for - separating the cost of dimension from the cost of conditioning,
and then trying the two standard fixes and reporting where each one stops.

Four things get built.

  1. **The estimator, checked against an exact number for the first time.** Day 1
     built the full transition matrix on a discretised state space so detailed
     balance could be a residual. The same matrix gives the exact IACT: `P` is
     reversible, so `D^(1/2) P D^(-1/2)` is symmetric, and expanding a centred
     `f` in its eigenbasis gives `IACT = 1 + 2 sum_j w_j lam_j / (1 - lam_j)` in
     closed form - no truncation, no window. Geyer's estimator has been reporting
     these numbers for two days with nothing to check it against.

     It holds up. Across proposal widths spanning IACT 10 to IACT 1632, and two
     test functions, the estimator lands within 6% of exact: 1728 vs 1632, 228.0
     vs 228.0, 16.1 vs 15.4. The worst case is `f = x^2` at half-width 4, which
     comes in 9% *low* - and low is the direction that matters, because an
     underestimated IACT is an overstated ESS. Nothing here is bad enough to
     change a conclusion, but the sign of the error is worth knowing.

  2. **Dimension against conditioning.** Isotropic, at day 1's `l* = 2.3`: IACT
     4.4, 7.9, 17.7, 32.8, 84.1, 179.2 for `d` = 1, 2, 5, 10, 25, 50. Linear in
     `d`, around `3.5 d`, no surprise. At `d = 2` with the condition number
     climbing: 8.0, 15.6, 61.8, 310.6, 901.7 for kappa = 1, 4, 19, 100, 400.

     So two correlated parameters at kappa = 100 cost IACT 311, and fifty
     independent ones cost 179. Day 1's observation was not an artefact of the
     one covariance matrix it used, and it gets worse the further you push it.

     What I will not claim is an exponent. The isotropic runs step at
     `sqrt(lam_min)`, which is the rule that makes sense - the narrow direction
     is what limits you - and the measured acceptance rate says that rule is
     wrong: it climbs 0.371, 0.479, 0.538, 0.558, 0.564 as kappa grows, while day
     1 established the efficient band sits far below that. The sampler is being
     under-stepped by a known amount in a known direction, so the growth I
     measured (a local slope near kappa^0.8) is the growth of a mistuned sampler
     and not of the problem. The ordering against dimension survives that, since
     mistuning only hurts the conditioned side. The exponent does not.

  3. **Preconditioning, and it does not approximately work - it works exactly.**
     Proposal covariance proportional to `Sigma` gives IACT 8.0 at kappa 19, 100
     and 400 alike, which is the isotropic `d = 2` number to the digit. That is
     affine invariance being visible rather than argued: the preconditioned chain
     on the correlated target *is* the isotropic chain, relabelled.

     The part I expected to fail is estimating `Sigma` from a pilot run, since
     the pilot is the badly-mixing chain the preconditioner is meant to fix -
     circular, and at kappa = 400 a 20k pilot at IACT ~900 holds about 20
     independent samples. It gives IACT 7.7, 8.1, 7.8, indistinguishable from
     using the exact matrix, off a covariance estimate that is up to 9.8% wrong.
     A preconditioner needs the orientation and the scale, and those are cheap;
     it does not need an accurate covariance, and I had been assuming the two
     requirements were the same.

     Then the banana, which has no global linear rescaling because the direction
     of dependence rotates with position. Isotropic: IACT 402 at acceptance
     0.521. Preconditioned with the pilot covariance: IACT 270 at acceptance
     **0.097**. So the fix does buy 1.5x, and it buys it by proposing into a
     shape the target does not have and being rejected nine times in ten. On a
     Gaussian, preconditioning is exact and free. Here it is a modest gain
     purchased at an acceptance rate day 1 put well outside the efficient band,
     and the covariance error it leaves behind is slightly worse than the
     isotropic chain's (0.062 vs 0.053), not better.

  4. **The independence sampler, where the theory is sharp and turns out not to
     be predictive.** `q(y|x) = q(y)`, so this is the one sampler in the project
     that carries the Hastings term. The classical result is a clean dichotomy:
     if `pi/q <= M` everywhere the chain is uniformly ergodic with gap `1/M`, so
     IACT is bounded by about `2M - 1`; if `pi/q` is unbounded there is no bound
     at all. On `N(0, I_2)` with an `N(0, s^2 I)` proposal the threshold is exact
     and sits at `s = 1`.

     The bound is correct and usefully tight - IACT 1.3, 2.0, 5.1, 13.3 against
     bounds 1.9, 4.1, 11.5, 31.0, conservative by a near-constant 2.3x, so it
     gets the scaling right even though it is not tight. What it is not is
     predictive across the threshold. At `s = 0.8` no bound exists and the
     sampler runs at IACT 4.5 with correct moments. At `s = 1.2` the bound is 1.9
     and IACT is 1.3. Nothing happens at `s = 1`. The property is binary and the
     performance is continuous through it, and the actual collapse is at
     `s = 0.6` (IACT 164), well past the point where the theorem stopped
     applying. A theorem that flips at a place where the measurement does not is
     not a tuning rule, and I had been reading it as one.

     And the one to remember, on the Student-t target where a Gaussian proposal
     has `pi/q` unbounded at every `s`. Acceptance rates 0.837, 0.556, 0.222 for
     `s` = 1, 2, 4. Errors in the variance against the true value of 3: 1.491,
     0.804, 0.490. They run in **opposite** directions. The setting that accepts
     84% of its proposals is the one that gets the variance 50% wrong, because it
     never proposes into the tail that carries the variance and instead sits
     still - its longest stuck run is 1034 steps. Ranking these three by
     acceptance rate reverses the ranking by correctness.

     Day 1 found a broken sampler whose mean was right and whose variance was
     16% high, and drew the moral that a first moment cannot detect an even bias.
     This is the same moral about the other diagnostic: acceptance rate is a
     property of the proposal, not evidence about the answer. Worth adding that
     the IACT estimator does not catch it either - it reports 24.5 on the chain
     with the 1034-step stall, because a rare long excursion is not what an
     autocorrelation sum is measuring.

What today is not: any statement about multimodality. Every target here is
unimodal or nearly so, and the stuck runs in [4] are tail behaviour rather than a
chain trapped in the wrong mode. Day 3 is HMC, where the gradient is supposed to
make the dimension scaling in [2] better, and day 4 is where the diagnostics get
scored on failures they are sold as detecting.

Run: `python day2_mixing.py`
"""

import numpy as np

from day1_metropolis import (
    CorrelatedGaussian,
    IIDGaussian,
    autocorrelation,
    detailed_balance_residual,
    discrete_metropolis_kernel,
    ess,
    integrated_act,
    random_walk_metropolis,
)


# --- exact IACT from the transition matrix -----------------------------------


def exact_iact(pi, P, f):
    """IACT of `f` under a reversible chain at stationarity, from the spectrum.

    Day 1 built the full kernel on a discretised state space so detailed balance
    could be a residual instead of an argument. The same matrix gives the exact
    autocorrelation time, which is what the estimator has been reporting without
    anything to check it against.

    `P` is reversible with respect to `pi`, so `D^{1/2} P D^{-1/2}` is symmetric
    for `D = diag(pi)`. Diagonalise that, expand `f - E[f]` in the eigenbasis,
    and `rho_k = sum_j w_j lambda_j^k` with weights `w_j` that are the squared
    coefficients normalised to sum to 1. Then

        IACT = 1 + 2 sum_{k>=1} rho_k = 1 + 2 sum_j w_j lambda_j / (1 - lambda_j)

    in closed form, no truncation and no window. The eigenvalue 1 carries weight
    0 because `f` is centred, so the sum is finite.
    """
    s = np.sqrt(pi)
    A = (s[:, None] * P) / s[None, :]
    A = 0.5 * (A + A.T)  # symmetric up to floating point; enforce it
    lam, V = np.linalg.eigh(A)

    g = f - pi @ f
    # coefficients of g in the pi-weighted inner product
    c = V.T @ (s * g)
    w = c**2
    var = w.sum()
    if var <= 0:
        return np.inf
    w = w / var

    # drop the stationary direction; it holds ~0 weight for centred f anyway
    keep = lam < 1.0 - 1e-12
    return 1.0 + 2.0 * float(np.sum(w[keep] * lam[keep] / (1.0 - lam[keep])))


# --- preconditioned and independence proposals -------------------------------


def preconditioned_metropolis(target, x0, n_steps, chol, rng):
    """Random-walk Metropolis with proposal covariance `chol @ chol.T`.

    Identical to day 1's sampler except that the proposal increment is
    `chol @ z` rather than `scale * z`. Still symmetric - a Gaussian centred at
    the current point with a fixed covariance is symmetric whatever that
    covariance is - so the acceptance ratio still omits the Hastings term and is
    still correct. Day 1's broken sampler made the covariance depend on the
    state, which is a different thing and is what actually breaks symmetry.
    """
    d = len(x0)
    x = np.array(x0, float)
    logp = target.log_density(x)
    chain = np.empty((n_steps, d))
    n_accept = 0

    normals = rng.standard_normal((n_steps, d))
    uniforms = rng.random(n_steps)

    for t in range(n_steps):
        y = x + chol @ normals[t]
        logq = target.log_density(y)
        if np.log(uniforms[t]) < logq - logp:
            x, logp = y, logq
            n_accept += 1
        chain[t] = x

    return chain, n_accept / n_steps


def independence_sampler(target, x0, n_steps, prop_mean, prop_chol, rng):
    """Metropolis-Hastings with a proposal that ignores the current state.

    `q(y|x) = q(y)`, so the Hastings ratio does not cancel and the acceptance
    probability is `min(1, [pi(y) q(x)] / [pi(x) q(y)])`. This is the one place
    in the project where the correction term is actually carried, and it is
    carried because the proposal is as asymmetric as a proposal can be.

    The theory here is unusually sharp: if `pi(x) / q(x) <= M` for all `x`, the
    chain is uniformly ergodic with spectral gap at least `1/M`, so IACT is
    bounded by roughly `2M - 1`. If the ratio is unbounded - proposal tails
    lighter than the target's - there is no such bound and the chain sticks. The
    run below measures both sides of that.
    """
    d = len(x0)
    prop_mean = np.asarray(prop_mean, float)
    prec = np.linalg.inv(prop_chol @ prop_chol.T)

    def log_q(z):
        r = z - prop_mean
        return -0.5 * r @ prec @ r

    x = np.array(x0, float)
    logp, logqx = target.log_density(x), log_q(x)
    chain = np.empty((n_steps, d))
    n_accept = 0

    normals = rng.standard_normal((n_steps, d))
    uniforms = rng.random(n_steps)

    for t in range(n_steps):
        y = prop_mean + prop_chol @ normals[t]
        logpy, logqy = target.log_density(y), log_q(y)
        if np.log(uniforms[t]) < (logpy - logp) + (logqx - logqy):
            x, logp, logqx = y, logpy, logqy
            n_accept += 1
        chain[t] = x

    return chain, n_accept / n_steps


class StudentT2D:
    """2d Student-t, `nu` degrees of freedom, zero mean, identity scale.

    Here only as an independence-sampler target: it has polynomial tails, so a
    Gaussian proposal has `pi/q` unbounded and the theory above has nothing to
    say. Moments exist for `nu > 2` and the covariance is `nu/(nu-2) I`.
    """

    def __init__(self, nu=3.0):
        self.nu, self.d, self.name = nu, 2, "student-t-2d"

    def log_density(self, x):
        return -0.5 * (self.nu + self.d) * np.log1p(x @ x / self.nu)

    def mean(self):
        return np.zeros(self.d)

    def cov(self):
        return (self.nu / (self.nu - 2.0)) * np.eye(self.d)


class Banana:
    """Rosenbrock-style banana: `x1 ~ N(0, 100)`, `x2 | x1 ~ N(b(x1^2 - 100), 1)`.

    A target with no global linear rescaling that fixes it - the direction of
    strong dependence rotates with position. It is here to be the case where the
    day's fix does not work, and its moments are still closed form:
    `Var(x1) = 100`, `E[x2] = 0`, `Var(x2) = 1 + b^2 Var(x1^2) = 1 + 2 b^2 100^2`.
    """

    def __init__(self, b=0.03):
        self.b, self.d, self.name = b, 2, "banana"

    def log_density(self, x):
        return -0.5 * (x[0] ** 2 / 100.0 + (x[1] - self.b * (x[0] ** 2 - 100.0)) ** 2)

    def mean(self):
        return np.zeros(2)

    def cov(self):
        v2 = 1.0 + (self.b**2) * 2.0 * 100.0**2
        # Cov(x1, x2) = b E[x1^3] = 0 by symmetry
        return np.array([[100.0, 0.0], [0.0, v2]])


def anisotropic_cov(d, kappa, rng):
    """Random `d x d` covariance with condition number exactly `kappa`."""
    evals = np.logspace(0, np.log10(kappa), d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q @ np.diag(evals) @ Q.T


# --- run ---------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("day 2 - what actually costs mixing, and two fixes that only sometimes work")
    print("=" * 78)

    # 1. the estimator, against an exact number for the first time
    print("\n[1] Geyer IACT vs the exact IACT from the kernel spectrum")
    print("    241-point discretised N(0,1); f(x) = x and f(x) = x^2")
    grid = np.linspace(-6, 6, 241)
    log_pi = -0.5 * grid**2
    pi = np.exp(log_pi - log_pi.max())
    pi /= pi.sum()
    for hw in (1, 4, 20):
        P = discrete_metropolis_kernel(log_pi, hw)
        assert detailed_balance_residual(pi, P) < 1e-12
        for fname, f in (("x", grid), ("x^2", grid**2)):
            tau = exact_iact(pi, P, f)
            # simulate the same finite-state chain and estimate it
            rng = np.random.default_rng(1000 + hw)
            n = 400_000
            cdf = np.cumsum(P, axis=1)
            u = rng.random(n)
            idx = np.empty(n, dtype=int)
            s = int(np.searchsorted(np.cumsum(pi), rng.random()))
            for t in range(n):
                s = int(np.searchsorted(cdf[s], u[t]))
                idx[t] = s
            hat = integrated_act(f[idx])
            print(
                f"    half_width={hw:3d}  f={fname:3s}  exact={tau:9.2f}"
                f"   estimated={hat:9.2f}   ratio={hat/tau:5.2f}"
            )

    # 2. dimension vs conditioning, the thing day 1 ended on
    print("\n[2] what costs more: dimension, or correlation at fixed dimension")
    print("    every run at its own optimal isotropic scale l/sqrt(d), l=2.3")
    n_steps = 120_000
    burn = n_steps // 10
    print("\n    isotropic, growing d")
    for d in (1, 2, 5, 10, 25, 50):
        tgt = IIDGaussian(d)
        rng = np.random.default_rng(4242 + d)
        ch, acc = random_walk_metropolis(
            tgt, np.zeros(d), n_steps, 2.3 / np.sqrt(d), rng
        )
        iact = max(integrated_act(ch[burn:, k]) for k in range(min(d, 5)))
        print(f"      d={d:3d}  kappa=  1  acc={acc:.3f}  IACT={iact:8.1f}")

    print("\n    d=2 fixed, growing condition number")
    kappa_rows = []
    for kappa in (1, 4, 19, 100, 400):
        rng0 = np.random.default_rng(7)
        sigma = anisotropic_cov(2, kappa, rng0) if kappa > 1 else np.eye(2)
        tgt = CorrelatedGaussian([0.0, 0.0], sigma)
        rng = np.random.default_rng(99)
        # isotropic proposal is limited by the *narrow* direction
        scale = 2.3 / np.sqrt(2) * np.sqrt(np.linalg.eigvalsh(sigma).min())
        ch, acc = random_walk_metropolis(tgt, np.zeros(2), n_steps, scale, rng)
        iact = max(integrated_act(ch[burn:, k]) for k in range(2))
        kappa_rows.append((kappa, acc, iact))
        print(f"      d=  2  kappa={kappa:3d}  acc={acc:.3f}  IACT={iact:8.1f}")

    print("\n    IACT ratio vs kappa ratio (is the cost linear in kappa?)")
    base = kappa_rows[0][2]
    for kappa, _, iact in kappa_rows:
        print(f"      kappa={kappa:4d}  IACT/IACT(1)={iact/base:8.2f}")

    # 3. fix one: precondition with the target covariance
    print("\n[3] fix 1 - proposal covariance proportional to the target's")
    print("    exact Sigma, then Sigma estimated from a pilot run")
    for kappa in (19, 100, 400):
        rng0 = np.random.default_rng(7)
        sigma = anisotropic_cov(2, kappa, rng0)
        tgt = CorrelatedGaussian([0.0, 0.0], sigma)

        rng = np.random.default_rng(99)
        chol = np.linalg.cholesky(sigma) * (2.3 / np.sqrt(2))
        ch, acc = preconditioned_metropolis(tgt, np.zeros(2), n_steps, chol, rng)
        iact_exact = max(integrated_act(ch[burn:, k]) for k in range(2))

        # pilot: a short badly-mixing isotropic run, then use its sample covariance
        rng = np.random.default_rng(1234)
        scale = 2.3 / np.sqrt(2) * np.sqrt(np.linalg.eigvalsh(sigma).min())
        pilot, _ = random_walk_metropolis(tgt, np.zeros(2), 20_000, scale, rng)
        shat = np.cov(pilot[2000:].T)
        rng = np.random.default_rng(99)
        cholhat = np.linalg.cholesky(shat) * (2.3 / np.sqrt(2))
        ch2, acc2 = preconditioned_metropolis(tgt, np.zeros(2), n_steps, cholhat, rng)
        iact_pilot = max(integrated_act(ch2[burn:, k]) for k in range(2))

        rel = np.abs(shat - sigma).max() / np.abs(sigma).max()
        print(
            f"    kappa={kappa:4d}  exact-Sigma IACT={iact_exact:7.1f} (acc={acc:.3f})"
            f"   pilot-Sigma IACT={iact_pilot:7.1f} (acc={acc2:.3f})"
            f"   |Sigma_hat - Sigma|/|Sigma|={rel:.3f}"
        )

    print("\n    the same fix on a target with no global linear rescaling")
    ban = Banana(0.03)
    rng = np.random.default_rng(5)
    scale = 2.3 / np.sqrt(2) * 1.0
    ch, acc = random_walk_metropolis(ban, np.zeros(2), n_steps, scale, rng)
    iact_iso = max(integrated_act(ch[burn:, k]) for k in range(2))
    rng = np.random.default_rng(5)
    pilot, _ = random_walk_metropolis(ban, np.zeros(2), 40_000, scale, rng)
    cholhat = np.linalg.cholesky(np.cov(pilot[4000:].T)) * (2.3 / np.sqrt(2))
    rng = np.random.default_rng(5)
    ch2, acc2 = preconditioned_metropolis(ban, np.zeros(2), n_steps, cholhat, rng)
    iact_pre = max(integrated_act(ch2[burn:, k]) for k in range(2))
    err_iso = np.abs(np.cov(ch[burn:].T) - ban.cov()).max() / np.abs(ban.cov()).max()
    err_pre = np.abs(np.cov(ch2[burn:].T) - ban.cov()).max() / np.abs(ban.cov()).max()
    print(
        f"      banana   isotropic  IACT={iact_iso:7.1f} (acc={acc:.3f})"
        f"  rel cov err={err_iso:.3f}"
    )
    print(
        f"      banana   precond    IACT={iact_pre:7.1f} (acc={acc2:.3f})"
        f"  rel cov err={err_pre:.3f}"
    )

    # 4. fix two: the independence sampler, and its cliff
    print("\n[4] fix 2 - the independence sampler, where pi/q decides everything")
    tgt = IIDGaussian(2)
    print("    target N(0, I_2); Gaussian proposal N(0, s^2 I), sweeping s")
    print("    sup pi/q is finite iff s > 1, and then M = s^d, bound IACT <~ 2M-1")
    for s in (0.6, 0.8, 1.0, 1.2, 1.6, 2.5, 4.0):
        rng = np.random.default_rng(31)
        ch, acc = independence_sampler(
            tgt, np.zeros(2), n_steps, np.zeros(2), s * np.eye(2), rng
        )
        iact = max(integrated_act(ch[burn:, k]) for k in range(2))
        merr = np.abs(ch[burn:].mean(axis=0)).max()
        verr = np.abs(ch[burn:].var(axis=0) - 1.0).max()
        bound = 2.0 * s**2 - 1.0 if s > 1.0 else np.inf
        bstr = f"{bound:8.1f}" if np.isfinite(bound) else "     inf"
        print(
            f"      s={s:4.1f}  acc={acc:.3f}  IACT={iact:8.1f}  bound={bstr}"
            f"   |mean|={merr:.4f}  |var-1|={verr:.4f}"
        )

    print("\n    heavy-tailed target, Gaussian proposal: pi/q unbounded whatever s")
    tgt = StudentT2D(3.0)
    for s in (1.0, 2.0, 4.0):
        rng = np.random.default_rng(31)
        ch, acc = independence_sampler(
            tgt, np.zeros(2), n_steps, np.zeros(2), s * np.eye(2), rng
        )
        iact = max(integrated_act(ch[burn:, k]) for k in range(2))
        verr = np.abs(ch[burn:].var(axis=0) - 3.0).max()
        longest = 0
        run = 1
        for k in range(1, len(ch)):
            run = run + 1 if ch[k, 0] == ch[k - 1, 0] else 1
            longest = max(longest, run)
        print(
            f"      s={s:4.1f}  acc={acc:.3f}  IACT={iact:8.1f}"
            f"  longest stuck run={longest:6d}  |var-3|={verr:.3f}"
        )

    print("\ndone.")
