"""Day 3 - the SoftAbs metric, for the Hessian the funnel has and not the one it has on average.

Day 2's metric was the Hessian of U averaged over x given v, diagonal, closed
form, and its two implicit solves were triangular. The Hessian itself has the
x_i e^-v cross terms, and its Schur complement in v is 1/sigma^2 - e^-v |x|^2 / 2,
so it is indefinite wherever e^-v |x|^2 > 2/sigma^2. Betancourt's SoftAbs takes
the eigenvalues lambda of that Hessian and uses lambda coth(alpha lambda): |lambda|
where alpha |lambda| is large, 1/alpha where it is small. The derivative of the
metric along q_k is Q (Q' (dHess/dq_k) Q o J) Q' with J the divided differences
of softabs over the eigenvalues, all closed form here, so the gradient of H can
be checked against finite differences. The solves are now genuinely implicit.
Same funnel, sigma_v = 3, d = 3, same seeds and cost counted in calls to dH/dq.

1. The Hessian is indefinite on 89.1% of draws from p, against exp(-1/9) =
   89.5% from the derivation, and the share is the same in the neck, 88.9%,
   and in the mouth, 88.4%: e^-v |x|^2 is chi^2_2 under p whatever v is, so
   this is not a neck property. The most negative eigenvalue on 20000 draws is
   -6.9, the median negative one -0.18. SoftAbs at alpha >= 10 gives x a mass
   within 3% to 6% of the Fisher metric's e^-v at the median, and gives v a
   mass that runs from 0.19 to 3.2 times the Fisher metric's constant across
   the draws. At alpha = 1 the floor of 1 is above e^-v over most of the mouth
   and the x mass is 1.5x the Fisher's at the median and 44x at the 90th
   percentile. dH/dq in closed form matches central differences of H to 3.9e-8
   relative at alpha = 100 and 6.3e-7 at 1000 and with the log det off.

2. The fixed point stops converging, which is day 2's question answered on the
   metric where it has something to solve, and it is alpha that sets where. At
   alpha = 1 it is day 1 again: 0 of 50 trajectories fail up to eps = 0.2 at
   3.6 to 7.1 iterations a solve. At alpha = 10 it fails on 13 of 50 at
   eps = 0.1 and 34 at 0.2. At alpha = 100 it fails on 6 of 50 at eps = 0.02,
   34 at 0.05 and 47 at 0.1, and at 1000 on 15 of 50 at eps = 0.02. It is the
   position solve that gives up, 44 of the 47 and 15 of the 15, and the steps
   that do converge carry |dH| of 3e-5 at eps = 0.02, so it is the solve and
   not the integrator, as on day 1. The map q -> q0 + eps (G^-1(q0) + G^-1(q)) p / 2
   contracts when eps |dG^-1/dq| |p| is small, and dG/dq carries the divided
   differences of softabs, which are of order alpha wherever an eigenvalue is
   near 0, which by point 1 is most of the target.

3. Every divergence in the SoftAbs chains is a solver failure. At eps = 0.2 and
   L = 10, 2000 steps: 616 of 616 at alpha = 10, 991 of 991 at 100, 1433 of
   1433 at 1000, with acceptance 0.645, 0.471, 0.250. The best SoftAbs row is
   alpha = 10, IACT(v) 9.95 and 86 effective samples per 100k gradients, at
   10.6 gradients a step. The Fisher metric at the same eps gives 427 and
   1753 at its own best eps, and centred HMC 157 and 462. So on this target
   SoftAbs through a fixed-point solve is 5x worse per gradient than the
   metric day 2 had for free and 2x worse than plain HMC, and the step size a
   converging solve allows, 0.02 at alpha = 100, gives IACT(v) 240 and 6
   effective samples per 100k gradients. The neck share is 6.28% and 7.78% at
   alpha = 10 and 100 against p's 8.28%, and 16.1% at 1000 where the chain
   sits in the neck because the solve fails on the way out.

4. With alpha = 10 the sampler runs at every sigma_v, ratio 1.12, 0.88, 0.57
   of the true Var(v) at sigma_v = 1, 3, 6 and min v about 2.5 sigma down,
   with 27% to 44% of proposals lost to the solve. At alpha >= 100 and
   sigma_v = 6 it accepts 1 of 1200, the momentum solve failing on 1069, so
   the deeper the funnel the smaller alpha has to be, and the smaller alpha
   the less the metric is the Hessian.

5. The log det term switched off moves the target and nothing else. With the
   Fisher metric det G = e^-2v g_v, so exp(-U) sqrt(det G) has v ~ N(-9, 9),
   and the chain gives mean v -9.01, Var(v) 9.08 and 94.84% of its mass below
   v = -4.16 against 94.7% predicted, at the same acceptance, 0.994, and the
   same IACT, 3.04, as the chain on the right target. No diagnostic in this
   project's kit sees the difference. With SoftAbs at alpha = 10 the mean is
   -8.98 against -8.69 by reweighting 20000 draws from p with sqrt(det G),
   whose weights have an ESS of 20, so that check is rough.

Seven predictions written before the run. Four right, one half, two wrong.

- Right: the Hessian is indefinite on about 90% of p, the same in the neck
  and the mouth. 89.1%, 88.9% against 88.4%.
- Right: closed-form dH/dq within 1e-6 relative of central differences. 6.3e-7
  at worst.
- Wrong: 5 to 10 iterations a solve at alpha = 100, eps = 0.2. 12.5 and 15.0,
  and 48 of 50 trajectories fail.
- Wrong: alpha = 100, eps = 0.2 puts the neck share within a point of 8.28%,
  min v below -9, divergences under 1%. 7.78% is, -8.17 and 50% are not.
- Right: with the log det off the Fisher chain has mean v -9 +- 0.5 and Var(v)
  about 9. -9.01 and 9.08.
- Right: IACT(v) at alpha = 1 at least 3x that at alpha = 100. 4.3x, though
  alpha = 100's 19.3 is made of rejected solves and not of mixing.
- Half: at least 10 gradients a step. 10.6 at alpha = 10, 9.3 at 100, 7.6 at
  1000, lower because a failed solve is cut short.

What this day does not settle is whether the fault is the metric or the
solver. The fixed-point iteration is the one Girolami and Calderhead wrote
down and the one day 1 measured; a damped iteration or a Newton step on the
position solve might take eps = 0.2 at alpha = 100, and that is the only way
SoftAbs earns a place in day 4's SMC move step. tol is 1e-8 throughout, which
day 1 says an acceptance rate cannot check.

NumPy only. Fixed seeds. About 7.5 minutes, most of it failed solves.
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcmc-from-scratch"))
from day1_generalised_leapfrog import fixed_point  # noqa: E402
from day2_fisher_funnel import BLOWUP, BURN, NECK, FisherFunnel, divergences, energy, phi, rmhmc  # noqa: E402
from day1_metropolis import integrated_act  # noqa: E402
from day3_hmc import Funnel, hmc  # noqa: E402

BAD = (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError)  # a solve that ran off


def softabs(lam, alpha):
    """lambda coth(alpha lambda): |lambda| far from 0, 1/alpha at 0, never below 1/alpha."""
    t = alpha * lam
    small = np.abs(t) < 1e-4
    safe = np.where(small, 1.0, t)
    return np.where(small, 1.0 / alpha + alpha * lam * lam / 3.0, lam / np.tanh(safe))


def softabs_prime(lam, alpha):
    """d/dlambda of lambda coth(alpha lambda). Zero at 0, sign(lambda) far from it."""
    t = alpha * lam
    small = np.abs(t) < 1e-4
    safe = np.clip(np.where(small, 1.0, t), -30.0, 30.0)
    return np.where(small, 2.0 * alpha * lam / 3.0, 1.0 / np.tanh(safe) - safe / np.sinh(safe) ** 2)


class SoftAbsFunnel:
    """The funnel (x_1..x_n, v) with G = Q softabs(Lambda) Q', Hessian of U = Q Lambda Q'.

    U = v^2 / (2 sigma^2) + e^-v |x|^2 / 2 + n v / 2. Its Hessian has e^-v on the
    x diagonal, -x_i e^-v across, and 1/sigma^2 + e^-v |x|^2 / 2 for v. The Schur
    complement of the x block is 1/sigma^2 - e^-v |x|^2 / 2, so the Hessian is
    indefinite wherever e^-v |x|^2 > 2 / sigma^2. SoftAbs replaces each eigenvalue
    by lambda coth(alpha lambda), which is |lambda| when alpha |lambda| is large and
    1/alpha when it is small. The derivative of G along q_k is Q (M_k o J) Q' with
    M_k = Q' (dHess/dq_k) Q and J the divided differences of softabs over the
    eigenvalues, softabs' on the diagonal.

    logdet=False drops the log det G / 2 term from H and its trace term from the
    gradient, which is the ablation the plan asked for.
    """

    def __init__(self, d=3, sigma_v=3.0, alpha=100.0, logdet=True):
        self.d, self.n, self.sigma_v, self.alpha, self.logdet = d, d - 1, sigma_v, alpha, logdet
        self.s2inv = 1.0 / sigma_v**2

    def potential(self, q):
        x, v = q[:-1], q[-1]
        return 0.5 * v * v * self.s2inv + 0.5 * math.exp(-v) * (x @ x) + 0.5 * self.n * v

    def grad_u(self, q):
        x, v = q[:-1], q[-1]
        ev = math.exp(-v)
        out = np.empty(self.d)
        out[:-1] = ev * x
        out[-1] = v * self.s2inv - 0.5 * ev * (x @ x) + 0.5 * self.n
        return out

    def hessian(self, q):
        x, v = q[:-1], q[-1]
        ev = math.exp(-v)
        h = np.zeros((self.d, self.d))
        h[np.arange(self.n), np.arange(self.n)] = ev
        h[:-1, -1] = h[-1, :-1] = -ev * x
        h[-1, -1] = self.s2inv + 0.5 * ev * (x @ x)
        return h

    def dhessian(self, q):
        """dHess / dq_k for each k, shape (d, d, d), closed form."""
        x, v = q[:-1], q[-1]
        ev = math.exp(-v)
        n = self.n
        out = np.zeros((self.d, self.d, self.d))
        for k in range(n):
            out[k, k, -1] = out[k, -1, k] = -ev
            out[k, -1, -1] = ev * x[k]
        out[-1, np.arange(n), np.arange(n)] = -ev
        out[-1, :-1, -1] = out[-1, -1, :-1] = ev * x
        out[-1, -1, -1] = -0.5 * ev * (x @ x)
        return out

    def metric(self, q):
        lam, Q = np.linalg.eigh(self.hessian(q))
        return lam, Q, softabs(lam, self.alpha)

    def g(self, q):
        lam, Q, s = self.metric(q)
        return (Q * s) @ Q.T

    def ginv(self, q):
        lam, Q, s = self.metric(q)
        return (Q / s) @ Q.T

    def draw_p(self, q, z):
        lam, Q, s = self.metric(q)
        return (Q * np.sqrt(s)) @ (Q.T @ z)

    def hamiltonian(self, q, p):
        lam, Q, s = self.metric(q)
        w = Q.T @ p
        h = self.potential(q) + 0.5 * np.sum(w * w / s)
        if self.logdet:
            h += 0.5 * np.sum(np.log(s))
        return h

    def dh_dq(self, q, p):
        lam, Q, s = self.metric(q)
        sp = softabs_prime(lam, self.alpha)
        diff = lam[:, None] - lam[None, :]
        close = np.abs(diff) < 1e-10 * (1.0 + np.max(np.abs(lam)))
        J = np.where(close, sp[:, None], (s[:, None] - s[None, :]) / np.where(close, 1.0, diff))
        w = (Q.T @ p) / s
        out = self.grad_u(q)
        for k, dh in enumerate(self.dhessian(q)):
            MJ = (Q.T @ dh @ Q) * J
            out[k] -= 0.5 * (w @ MJ @ w)
            if self.logdet:
                out[k] += 0.5 * np.sum(np.diag(MJ) / s)
        return out


def gl_step(model, q, p, eps, tol, max_iter, count):
    """Day 1's generalised leapfrog with a full metric. count[0] is dH/dq calls, count[1] metric evaluations."""
    def grad(qq, pp):
        count[0] += 1
        return model.dh_dq(qq, pp)

    def ginv(qq):
        count[1] += 1
        return model.ginv(qq)

    p_half, k1, ok1 = fixed_point(lambda ph: p - 0.5 * eps * grad(q, ph), p, tol, max_iter)
    ok1 = ok1 and bool(np.isfinite(p_half).all())
    if not ok1:
        return q, p, k1, 0, False, "p"
    ginv0 = ginv(q)
    q_new, k2, ok2 = fixed_point(lambda qn: q + 0.5 * eps * (ginv0 + ginv(qn)) @ p_half, q, tol, max_iter)
    ok2 = ok2 and bool(np.isfinite(q_new).all())
    if not ok2:
        return q, p, k1, k2, False, "q"
    p_new = p_half - 0.5 * eps * grad(q_new, p_half)
    return q_new, p_new, k1, k2, bool(np.isfinite(p_new).all()), ""


def softabs_chain(model, n_steps, eps, n_leap, rng, tol=1e-8, max_iter=50):
    """RMHMC with the SoftAbs metric, p ~ N(0, G(q)) redrawn every step.

    A solve that does not converge is a rejected proposal with dH = inf, counted
    apart. Returns chain, acceptance, dH, gradient calls per step, solver failures.
    """
    q = np.zeros(model.d)
    chain = np.empty((n_steps, model.d))
    dH = np.empty(n_steps)
    noise = rng.standard_normal((n_steps, model.d))
    unif = np.log(rng.random(n_steps))
    acc = 0
    fails = {"p": 0, "q": 0}
    count = [0, 0]
    for t in range(n_steps):
        try:
            p = model.draw_p(q, noise[t])
            h0 = model.hamiltonian(q, p)
            qn, pn, ok, which = q, p, True, ""
            for _ in range(n_leap):
                qn, pn, _, _, ok, which = gl_step(model, qn, pn, eps, tol, max_iter, count)
                if not ok:
                    break
            if ok:
                delta = model.hamiltonian(qn, pn) - h0
            else:
                delta = math.inf
                fails[which or "p"] += 1
        except BAD:
            delta = math.inf
        dH[t] = delta
        if math.isfinite(delta) and unif[t] < -delta:
            q = qn
            acc += 1
        chain[t] = q
    return chain, acc / n_steps, dH, count[0] / (n_steps * n_leap), f"{fails['p'] + fails['q']} (p {fails['p']}, q {fails['q']})"


def fisher_step_off(x, v, px, pv, eps, s2inv, gv, shift):
    """Day 2's step by hand with the log det term removed: U's n v / 2 no longer cancels, so dH/dv gains n / 2."""
    ev = math.exp(-v)
    px = [a - 0.5 * eps * ev * b for a, b in zip(px, x)]
    pp = sum(a * a for a in px)
    pv = pv - 0.5 * eps * (v * s2inv - 0.5 * ev * sum(b * b for b in x) + 0.5 * pp / ev + shift)
    v_new = v + eps * pv / gv
    scale = 0.5 * eps * (1.0 / ev + math.exp(v_new))
    x = [b + scale * a for a, b in zip(px, x)]
    ev = math.exp(-v_new)
    pv = pv - 0.5 * eps * (v_new * s2inv - 0.5 * ev * sum(b * b for b in x) + 0.5 * pp / ev + shift)
    px = [a - 0.5 * eps * ev * b for a, b in zip(px, x)]
    return x, v_new, px, pv


def fisher_chain_off(model, n_steps, eps, n_leap, rng):
    """Day 2's sampler with H = U + p' G^-1 p / 2 and no log det. Targets exp(-U) sqrt(det G)."""
    s2inv, gv, n = 1.0 / model.sigma_v**2, model.gv, model.n
    shift = 0.5 * n
    x, v = [0.0] * n, 0.0
    chain = np.empty((n_steps, n + 1))
    dH = np.empty(n_steps)
    noise = rng.standard_normal((n_steps, n + 1))
    unif = np.log(rng.random(n_steps))
    acc = 0
    for t in range(n_steps):
        px = [math.exp(-0.5 * v) * z for z in noise[t, :n]]
        pv = math.sqrt(gv) * noise[t, n]
        h0 = energy(x, v, px, pv, s2inv, gv) + shift * v
        xn, vn, pxn, pvn = x, v, px, pv
        try:
            for _ in range(n_leap):
                xn, vn, pxn, pvn = fisher_step_off(xn, vn, pxn, pvn, eps, s2inv, gv, shift)
            delta = energy(xn, vn, pxn, pvn, s2inv, gv) + shift * vn - h0
        except BLOWUP:
            delta = math.inf
        dH[t] = delta
        if math.isfinite(delta) and unif[t] < -delta:
            x, v = xn, vn
            acc += 1
        chain[t, :n], chain[t, n] = x, v
    return chain, acc / n_steps, dH


def report(label, chain, acc, dH, grads=None, extra=""):
    v = chain[int(BURN * len(chain)):, -1]
    iact = integrated_act(v)
    line = (f"  {label}  acc {acc:.3f}  div {divergences(dH):5d}  mean v {v.mean():6.2f}  Var(v) {v.var():6.2f}"
            f"  min v {v.min():7.2f}  neck {100 * np.mean(v < NECK):5.2f}%  IACT(v) {iact:7.2f}")
    if grads is not None:
        ess = 0.9 * len(chain) / iact
        line += f"  grads/step {grads:5.1f}  ess/100k grads {1e5 * ess / (grads * len(chain) * 10):6.0f}"
    print(line + extra)
    return iact


def draw_p_funnel(rng, d, sigma_v):
    v = sigma_v * rng.standard_normal()
    return np.append(math.exp(0.5 * v) * rng.standard_normal(d - 1), v)


# ----------------------------------------------------------------------------


def main():
    np.seterr(all="ignore")
    started = time.time()
    d, sigma_v = 3, 3.0
    fisher = FisherFunnel(d)
    rng = np.random.default_rng(33)
    print(f"funnel d = {d}, sigma_v = {sigma_v}: P(v < {NECK}) = {100 * phi(NECK / sigma_v):.2f}%,"
          f" the hessian is indefinite where e^-v |x|^2 > 2 / sigma^2 = {2 / sigma_v**2:.3f},"
          f" which under p is chi^2_2 > {2 / sigma_v**2:.3f}: {100 * math.exp(-1 / sigma_v**2):.1f}%")

    print("\n== the hessian of U on 20000 draws from p, and what softabs makes of it ==")
    probe = SoftAbsFunnel(d, sigma_v)
    draws = [draw_p_funnel(rng, d, sigma_v) for _ in range(20_000)]
    lam_min = np.array([np.linalg.eigvalsh(probe.hessian(q))[0] for q in draws])
    vs = np.array([q[-1] for q in draws])
    ss = np.array([math.exp(-q[-1]) * (q[:-1] @ q[:-1]) for q in draws])
    print(f"  indefinite {100 * np.mean(lam_min < 0):.1f}%   by the schur complement {100 * np.mean(ss > 2 / sigma_v**2):.1f}%"
          f"   in the neck {100 * np.mean(lam_min[vs < NECK] < 0):.1f}%   in the mouth, v > 4.16: {100 * np.mean(lam_min[vs > -NECK] < 0):.1f}%")
    print(f"  most negative eigenvalue {lam_min.min():.3e}   median of the negative ones {np.median(lam_min[lam_min < 0]):.3e}")
    print("  G's diagonal against the fisher metric's, medians over the draws, and the smallest eigenvalue of G:")
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        m = SoftAbsFunnel(d, sigma_v, alpha)
        rx, rv, smin = [], [], []
        for q in draws[:4000]:
            G = m.g(q)
            rx.append(G[0, 0] / math.exp(-q[-1])), rv.append(G[-1, -1] / fisher.gv)
            smin.append(np.linalg.eigvalsh(G)[0])
        print(f"    alpha {alpha:6.0f}   G_xx / e^-v {np.median(rx):6.3f} ({np.percentile(rx, 10):.3f} to {np.percentile(rx, 90):.3f})"
              f"   G_vv / g_v {np.median(rv):6.3f} ({np.percentile(rv, 10):.3f} to {np.percentile(rv, 90):.3f})"
              f"   min eig {np.min(smin):.2e}  median {np.median(smin):.2e}")

    print("\n== dH/dq in closed form against central differences of H, 30 draws, h = 1e-5 ==")
    for alpha, logdet in ((100.0, True), (1000.0, True), (100.0, False)):
        m = SoftAbsFunnel(d, sigma_v, alpha, logdet)
        worst = 0.0
        for q in draws[:30]:
            p = m.draw_p(q, rng.standard_normal(d))
            g = m.dh_dq(q, p)
            fd = np.empty(d)
            for k in range(d):
                e = np.zeros(d)
                e[k] = 1e-5
                fd[k] = (m.hamiltonian(q + e, p) - m.hamiltonian(q - e, p)) / 2e-5
            worst = max(worst, np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd))))
        print(f"  alpha {alpha:6.0f}  logdet {str(logdet):5s}   max relative error {worst:.2e}")

    print("\n== the implicit solves: 50 draws from p, 20 steps, tol 1e-8, and which solve gives up first ==")
    print("   alpha    eps   p iters  q iters  grads/step  metric/step  failed  (p solve, q solve)   median |dH|")
    starts = [(q, None) for q in draws[:50]]
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        m = SoftAbsFunnel(d, sigma_v, alpha)
        for eps in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
            k1s, k2s, dh = [], [], []
            fails = {"p": 0, "q": 0}
            count = [0, 0]
            steps = 0
            for q0, _ in starts:
                p0 = m.draw_p(q0, np.random.default_rng(7).standard_normal(d))
                q, p, good = q0, p0, True
                try:
                    for _ in range(20):
                        q, p, k1, k2, ok, which = gl_step(m, q, p, eps, 1e-8, 50, count)
                        steps += 1
                        k1s.append(k1), k2s.append(k2)
                        if not ok:
                            good = False
                            fails[which or "p"] += 1
                            break
                    if good:
                        dh.append(abs(m.hamiltonian(q, p) - m.hamiltonian(q0, p0)))
                except BAD:
                    fails["p"] += 1
            print(f"  {alpha:6.0f}   {eps:4.2f}   {np.mean(k1s):5.2f}    {np.mean(k2s):5.2f}     {count[0] / steps:5.2f}"
                  f"       {count[1] / steps:5.2f}     {fails['p'] + fails['q']:3d}/50   ({fails['p']:3d}, {fails['q']:3d})"
                  f"        {np.median(dh) if dh else np.nan:.2e}  ({len(dh)} finite)")

    n_steps, n_leap = 2000, 10
    print(f"\n== the chains: {n_steps} steps of L = {n_leap}, burn {int(100 * BURN)}%, ess per 100k gradient calls ==")
    print("  centred hmc from the mcmc project, one gradient a step:")
    base = {}
    for eps in (0.1, 0.2, 0.4):
        ch, acc, dH, _ = hmc(Funnel(d), np.zeros(d), n_steps, eps, n_leap, np.random.default_rng(11))
        base[eps] = report(f"    eps {eps:4.2f}", ch, acc, dH, grads=1.0)
    print("  fisher metric from day 2, two gradients a step by hand:")
    for eps in (0.1, 0.2, 0.4):
        ch, acc, dH = rmhmc(fisher, n_steps, eps, n_leap, np.random.default_rng(11))
        report(f"    eps {eps:4.2f}", ch, acc, dH, grads=2.0)
    print("  softabs:")
    for alpha, eps in ((1.0, 0.2), (10.0, 0.2), (100.0, 0.2), (1000.0, 0.2), (100.0, 0.02), (100.0, 0.05), (100.0, 0.1), (100.0, 0.4)):
        m = SoftAbsFunnel(d, sigma_v, alpha)
        ch, acc, dH, grads, fails = softabs_chain(m, n_steps, eps, n_leap, np.random.default_rng(11))
        report(f"    alpha {alpha:6.0f} eps {eps:4.2f}", ch, acc, dH, grads=grads, extra=f"  solver failed {fails}")

    print("\n== the log det term switched off ==")
    print(f"  with the fisher metric det G = e^-2v g_v, so the chain should target v ~ N(-9, 9):"
          f" neck share {100 * phi((NECK + 9.0) / sigma_v):.1f}%, mean v -9.00")
    ch, acc, dH = fisher_chain_off(fisher, 20_000, 0.2, 20, np.random.default_rng(11))
    report("  fisher, log det off, 20k steps L 20", ch, acc, dH)
    ch, acc, dH = rmhmc(fisher, 20_000, 0.2, 20, np.random.default_rng(11))
    report("  fisher, log det on,  20k steps L 20", ch, acc, dH)
    for alpha in (10.0, 100.0):
        m = SoftAbsFunnel(d, sigma_v, alpha, logdet=False)
        ch, acc, dH, grads, fails = softabs_chain(m, n_steps, 0.2, n_leap, np.random.default_rng(11))
        report(f"  softabs alpha {alpha:4.0f}, log det off", ch, acc, dH, grads=grads, extra=f"  solver failed {fails}")
        m = SoftAbsFunnel(d, sigma_v, alpha, logdet=False)
        # the target the chain is actually sampling, by importance weights from p: E_p[sqrt(det G)] normalises it
        w = np.array([0.5 * np.sum(np.log(softabs(np.linalg.eigvalsh(m.hessian(q)), alpha))) for q in draws])
        w = np.exp(w - w.max())
        w /= w.sum()
        print(f"      exp(-U) sqrt(det G) by reweighting the 20000 draws: mean v {np.sum(w * vs):6.2f}"
              f"   neck {100 * np.sum(w * (vs < NECK)):5.1f}%   ess of the weights {1 / np.sum(w * w):.0f}")

    print("\n== alpha against the neck depth: sigma_v swept at eps = 0.2, L = 10, 1200 steps ==")
    for sig in (1.0, 3.0, 6.0):
        for alpha in (10.0, 100.0, 1000.0):
            m = SoftAbsFunnel(d, sig, alpha)
            ch, acc, dH, grads, fails = softabs_chain(m, 1200, 0.2, n_leap, np.random.default_rng(31))
            v = ch[120:, -1]
            print(f"  sigma_v {sig:3.1f} alpha {alpha:5.0f}   acc {acc:.3f}  div {divergences(dH):4d}  ratio {v.var() / sig**2:5.3f}"
                  f"  min v / sigma {v.min() / sig:6.2f}  IACT {integrated_act(v):6.2f}  grads/step {grads:5.1f}  failed {fails}")

    print(f"\n{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
