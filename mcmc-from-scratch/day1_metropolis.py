"""
Day 1 of MCMC.

The particle filter project ended by pointing here. Its last day built a
likelihood estimator and showed it was unbiased at every particle count, and an
unbiased likelihood estimator is exactly what pseudo-marginal MCMC consumes - but
that only means anything if the MCMC part is understood on its own, on targets
where the answer is known, rather than bolted onto a filter whose output I would
then have no way to check.

So the whole project runs on targets with closed-form moments. Every claim about
error is a claim against an exact number, the same discipline the grid filter
bought on the last project.

Today is the base case: random-walk Metropolis, why it is correct, and how badly
it is tuned by default. Three things get built.

  1. **The correctness argument, as a residual rather than an argument.** The
     state space is made finite - a discretised Gaussian on 241 grid points - so
     the transition kernel is a matrix that can be written down in full. Detailed
     balance stops being a line of algebra and becomes `max |pi_i P_ij -
     pi_j P_ji|`, which comes out at 1.6e-19 to 1.7e-18 depending on the proposal
     width, i.e. zero. Stationarity `|pi P - pi|` is 3.5e-18. The same matrix
     gives the exact relaxation time from its second eigenvalue, and that is the
     first honest mixing number in either project: 816.6 steps at a
     one-cell proposal against 8.4 at twenty cells, a factor of 97 from
     the tuning parameter alone, computed rather than estimated.

  2. **What the symmetry assumption is holding up.** Metropolis omits the
     proposal-density ratio because `q(y|x) = q(x|y)`. Making the proposal width
     depend on the current state breaks that and nothing else. The result is the
     one worth remembering: the mean stays correct (-0.0048 against 0) and the
     variance is 1.1649 against 1, 16% high, with `E|x|` at 0.8815 against
     `sqrt(2/pi) = 0.7979`. The bias is even in `x`, so the first moment is
     structurally protected and cannot detect it. A sampler checked only on its
     mean passes here while sampling the wrong distribution.

  3. **The 0.234 result, measured.** Sweep the proposal scale `l/sqrt(d)` for
     `d` in 1..50, three seeds each, and take the peak of ESS per step.

What came out, since the second and third of these are not what I went in with:

  - `l*` is 2.30 for every `d >= 2` on this grid - flat, immediately, against the
    asymptotic 2.38 - while the acceptance rate at the optimum is still falling
    at `d = 50`: 0.417, 0.369, 0.301, 0.276, 0.261, 0.255 for d = 1, 2, 5, 10,
    25, 50. So the quantity you set converges at `d = 2` and the quantity you are
    told to set it *by* is 9% off at `d = 50` and heading down slowly. Tuning to
    an acceptance target is the worse of the two rules at every dimension I can
    reach, and it is the one that gets quoted;
  - the efficiency peak is very flat. At `d = 50` every acceptance rate from
    0.153 to 0.320 is within 10% of peak ESS - a factor of two - so the three
    decimal places in "0.234" describe a property of the limit and not a
    tolerance anyone should be tuning into;
  - correlation costs more than dimension does at this size. `N(0, I_2)` runs at
    IACT 8.1; a 2d Gaussian with condition number 19 runs at 46.8, worse than
    `d = 50` isotropic, on a target with the same number of parameters. That is
    day 2's subject and it is why day 2 is about preconditioning and not only
    about dimension;
  - and the efficiency estimator's first version reported a completely stuck
    chain as the most efficient setting on the grid. Written up in
    `integrated_act`.

What today is not: any claim about mixing between separated modes. The mixture
target's moments come out fine at IACT 8.0 because its components overlap; a
mixture that actually traps the chain is what HMC and the diagnostics are for,
and asserting anything about it before day 4 would be guessing.

Run: `python day1_metropolis.py`
"""

import numpy as np


# --- targets with closed-form moments ---------------------------------------


class IIDGaussian:
    """`N(0, I_d)`. Moments known exactly, which is the point of using it."""

    def __init__(self, d):
        self.d = d
        self.name = f"iid-gaussian-d{d}"

    def log_density(self, x):
        return -0.5 * np.dot(x, x)

    def mean(self):
        return np.zeros(self.d)

    def cov(self):
        return np.eye(self.d)


class CorrelatedGaussian:
    """`N(mu, Sigma)` in 2d with a deliberate condition number."""

    def __init__(self, mu, sigma):
        self.mu = np.asarray(mu, float)
        self.sigma = np.asarray(sigma, float)
        self.d = len(self.mu)
        self.prec = np.linalg.inv(self.sigma)
        self.name = "correlated-gaussian"

    def log_density(self, x):
        r = x - self.mu
        return -0.5 * r @ self.prec @ r

    def mean(self):
        return self.mu

    def cov(self):
        return self.sigma


class GaussianMixture1D:
    """Two-component 1d mixture. Mean and variance in closed form.

    `p N(m0, s0^2) + (1-p) N(m1, s1^2)`; the variance is the law of total
    variance, `E[Var] + Var[E]`, and the second term is what a unimodal
    approximation throws away.
    """

    def __init__(self, p, m0, s0, m1, s1):
        self.p, self.m0, self.s0, self.m1, self.s1 = p, m0, s0, m1, s1
        self.d = 1
        self.name = "mixture-1d"

    def log_density(self, x):
        a = np.log(self.p) - 0.5 * ((x[0] - self.m0) / self.s0) ** 2 - np.log(self.s0)
        b = np.log(1 - self.p) - 0.5 * ((x[0] - self.m1) / self.s1) ** 2 - np.log(self.s1)
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    def mean(self):
        return np.array([self.p * self.m0 + (1 - self.p) * self.m1])

    def cov(self):
        em = self.mean()[0]
        second = self.p * (self.s0**2 + self.m0**2) + (1 - self.p) * (self.s1**2 + self.m1**2)
        return np.array([[second - em**2]])


# --- random-walk Metropolis --------------------------------------------------


def random_walk_metropolis(target, x0, n_steps, scale, rng, state_dependent=False):
    """Symmetric-proposal Metropolis. Returns the chain and the acceptance rate.

    `scale` is the standard deviation of the isotropic Gaussian proposal. The
    acceptance ratio is `min(1, pi(y)/pi(x))` with no proposal-density term,
    which is only correct because `q(y|x) = q(x|y)`.

    `state_dependent=True` deliberately breaks that: the proposal standard
    deviation becomes `scale * (1 + |x|_2 / 2)`, so `q` stops being symmetric
    while the acceptance ratio still omits the Hastings correction. It is here to
    be measured, not used.
    """
    d = len(x0)
    x = np.array(x0, float)
    logp = target.log_density(x)
    chain = np.empty((n_steps, d))
    n_accept = 0

    normals = rng.standard_normal((n_steps, d))
    uniforms = rng.random(n_steps)

    for t in range(n_steps):
        s = scale * (1.0 + np.linalg.norm(x) / 2.0) if state_dependent else scale
        y = x + s * normals[t]
        logq = target.log_density(y)
        if np.log(uniforms[t]) < logq - logp:
            x, logp = y, logq
            n_accept += 1
        chain[t] = x

    return chain, n_accept / n_steps


# --- autocorrelation and effective sample size -------------------------------


def autocorrelation(v):
    """Normalised autocorrelation function of a 1d series, by FFT.

    Undefined for a constant series, and that case is not hypothetical here - a
    proposal scale large enough to reject everything produces one. It is handled
    in `integrated_act` rather than by returning something plausible from here.
    """
    v = np.asarray(v, float)
    v = v - v.mean()
    n = len(v)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(v, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
    return acf / acf[0]


def integrated_act(v):
    """IACT via Geyer's initial positive sequence estimator.

    Sum the ACF in adjacent *pairs* and stop at the first non-positive pair. The
    pairing is what makes it work - for a reversible chain the pair sums are
    positive, so the truncation point is determined by the data rather than by a
    window length I would otherwise have to pick and then defend.

    A chain that never moved has zero sample variance, and a normalised ACF
    cannot tell that apart from zero autocorrelation - both are `0/0`. Left
    alone it comes back as `nan`, the pair loop breaks at once, and the estimator
    reports IACT 1: a stuck chain producing one independent sample per step.
    Infinity is the honest answer, and it is the difference between the sweep
    below finding the efficiency peak and finding the largest scale on the grid.
    """
    v = np.asarray(v, float)
    if not np.isfinite(v).all() or v.std() == 0.0:
        return np.inf
    acf = autocorrelation(v)
    n = len(acf)
    pairs = []
    for k in range(0, n - 1, 2):
        s = acf[k] + acf[k + 1]
        if s <= 0:
            break
        pairs.append(s)
    if not pairs:
        return 1.0
    # enforce the monotone-decreasing envelope Geyer's estimator also asks for
    for i in range(1, len(pairs)):
        pairs[i] = min(pairs[i], pairs[i - 1])
    return max(1.0, 2.0 * sum(pairs) - 1.0)


def ess(v):
    return len(v) / integrated_act(v)


# --- detailed balance as a numerical identity --------------------------------


def discrete_metropolis_kernel(log_pi, half_width):
    """Exact transition matrix of Metropolis on a finite state space.

    `log_pi` is the log target on `m` states. The proposal is uniform on the
    `2*half_width` neighbours within range; a proposal that falls off the end is
    a rejection, which keeps `q` symmetric on the states that exist. Returns the
    full `m x m` matrix, so detailed balance stops being an argument and becomes
    something with a residual.
    """
    m = len(log_pi)
    q = 1.0 / (2 * half_width)
    P = np.zeros((m, m))
    for i in range(m):
        for k in range(-half_width, half_width + 1):
            j = i + k
            if k == 0 or j < 0 or j >= m:
                continue
            P[i, j] = q * min(1.0, np.exp(log_pi[j] - log_pi[i]))
        P[i, i] = 1.0 - P[i].sum()
    return P


def detailed_balance_residual(pi, P):
    F = pi[:, None] * P
    return np.abs(F - F.T).max()


# --- run ---------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("day 1 - random-walk Metropolis, detailed balance, and the 0.234 claim")
    print("=" * 78)

    # 1. detailed balance on a state space small enough to write the kernel down
    print("\n[1] detailed balance, exactly")
    grid = np.linspace(-6, 6, 241)
    log_pi = -0.5 * grid**2
    pi = np.exp(log_pi - log_pi.max())
    pi /= pi.sum()
    for hw in (1, 4, 20):
        P = discrete_metropolis_kernel(log_pi, hw)
        res = detailed_balance_residual(pi, P)
        stat = np.abs(pi @ P - pi).max()
        evals = np.linalg.eigvals(P.T)
        slem = np.sort(np.abs(evals))[-2]
        print(
            f"    half_width={hw:3d}  |pi_i P_ij - pi_j P_ji|_max={res:.3e}"
            f"   |pi P - pi|_max={stat:.3e}   SLEM={slem:.6f}"
            f"   relax={1/(1-slem):8.1f}"
        )

    # 2. what the symmetry assumption is actually holding up
    print("\n[2] the same sampler with an asymmetric proposal and no Hastings term")
    target1 = IIDGaussian(1)
    rng = np.random.default_rng(20260902)
    for flag, label in ((False, "symmetric  "), (True, "state-dep  ")):
        ch, acc = random_walk_metropolis(
            target1, np.zeros(1), 400_000, 2.0, rng, state_dependent=flag
        )
        v = ch[:, 0]
        print(
            f"    {label} acc={acc:.3f}  mean={v.mean():+.4f} (true 0)"
            f"   var={v.var():.4f} (true 1)   E|x|={np.abs(v).mean():.4f} (true 0.7979)"
        )

    # 3. moments against closed form on targets where the answer is known
    print("\n[3] moments vs closed form, 200k steps each")
    targets = [
        (IIDGaussian(2), 2.0),
        (CorrelatedGaussian([1.0, -2.0], [[4.0, 3.6], [3.6, 4.0]]), 1.2),
        (GaussianMixture1D(0.35, -2.0, 0.7, 2.5, 1.2), 3.0),
    ]
    for tgt, scale in targets:
        rng = np.random.default_rng(7)
        ch, acc = random_walk_metropolis(tgt, np.zeros(tgt.d), 200_000, scale, rng)
        mhat, chat = ch.mean(axis=0), np.cov(ch.T).reshape(tgt.d, tgt.d)
        merr = np.abs(mhat - tgt.mean()).max()
        cerr = np.abs(chat - tgt.cov()).max()
        iact = max(integrated_act(ch[:, k]) for k in range(tgt.d))
        print(
            f"    {tgt.name:22s} acc={acc:.3f}  IACT={iact:7.1f}"
            f"  |mean err|={merr:.4f}  |cov err|={cerr:.4f}"
        )

    # 4. the 0.234 claim, measured
    print("\n[4] optimal scaling: sweep the proposal scale, find the efficiency peak")
    print("    proposal sd = l / sqrt(d); efficiency = ESS per 1000 steps")
    n_steps = 50_000
    seeds = (11, 12, 13)
    ls = np.array([0.5, 0.9, 1.3, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.2, 5.0, 6.5, 9.0])
    summary = []
    for d in (1, 2, 5, 10, 25, 50):
        tgt = IIDGaussian(d)
        rows = []
        for l in ls:
            accs, effs = [], []
            for seed in seeds:
                rng = np.random.default_rng(seed * 100 + d)
                ch, acc = random_walk_metropolis(
                    tgt, np.zeros(d), n_steps, l / np.sqrt(d), rng
                )
                burn = n_steps // 10
                accs.append(acc)
                effs.append(np.mean([ess(ch[burn:, k]) for k in range(min(d, 5))]))
            rows.append(
                (l, float(np.mean(accs)), 1000.0 * float(np.mean(effs)) / (n_steps - burn))
            )
        best = max(rows, key=lambda r: r[2])
        near = [r for r in rows if r[2] >= 0.9 * best[2]]
        summary.append((d, best, min(r[1] for r in near), max(r[1] for r in near)))
        print(
            f"    d={d:3d}  l*={best[0]:.2f}  acc*={best[1]:.3f}"
            f"  ESS/1k={best[2]:6.2f}   acc within 10% of peak: "
            f"[{min(r[1] for r in near):.3f}, {max(r[1] for r in near):.3f}]"
        )

    print("\n      d     l*    acc*   |  asymptotic: l*=2.38, acc*=0.234")
    for d, best, lo, hi in summary:
        print(f"    {d:3d}   {best[0]:5.2f}  {best[1]:.3f}  |  band [{lo:.3f}, {hi:.3f}]")

    print("\ndone.")
