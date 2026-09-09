"""Day 2 - the two gradients of the same ELBO, with their variances printed.

Day 1 had a closed form for everything, which is why it could check that the
ELBO gap is exactly `KL(q || p)` and that the mean-field fixed point misses the
variance by the conditional-over-marginal ratio. Nothing there needed a
gradient. Black-box VI is the version that does not get the closed form: it
estimates `grad ELBO` from samples of `q` and climbs. So the object of interest
stops being the bound and becomes the estimator of its gradient, and an
estimator has a variance whether or not anyone reports it.

Both estimators are unbiased for the same quantity. On the conjugate targets
from day 1 the true gradient is known exactly, so unbiasedness is a check rather
than a citation, and the whole comparison is variance.

What is measured.

1. Both are unbiased, and the check is against the exact gradient rather than
   against each other. Over 400 batches of 512 samples every coordinate of both
   estimators sits inside 1.8 standard errors of the truth. So the difference
   between them is not in the mean and there is nothing else to look at but
   variance.

2. Variance, and here the number I went in with was wrong by a lot. The claim I
   have read and repeated is that the reparameterisation gradient is orders of
   magnitude tighter. At d = 4 it is 1.7x on the mean coordinate and 3.1x on the
   log-scale one. The orders of magnitude are a statement about dimension and
   not about the estimators: 1.6x at d = 2, 3.4x at d = 4, 11.2x at d = 8, 38.1x
   at d = 16. The score function's variance is the one that grows, because
   `grad log q` gains coordinates while the factor multiplying it does not.

3. The standard recipe makes it worse here, which is the finding worth keeping.
   Rao-Blackwellisation for mean-field BBVI drops, from coordinate i's
   estimator, the `log q_j` terms for `j != i`: they are independent of
   `grad_i log q` under a factorised `q`, the product's expectation is zero, so
   dropping them is unbiased. It is also the only thing there is to drop on a
   correlated target, since `log p` does not factor. Measured, the dropped term
   takes the per-sample variance from 6.73 to 22.81, and adding a scalar control
   variate on top recovers only part of it - 1.28e-02 plain against 1.92e-02
   for the full recipe.

   The reason is one covariance. `Var(A - B) = Var(A) + Var(B) - 2Cov(A, B)`,
   and here `Var(B) = 16.29` against `Cov(A, B) = 16.19`, correlation 0.84, so
   `2Cov - Var(B) = +16.09` is what deleting B adds. The term being dropped was
   doing the job of a control variate. Unbiasedness was the only thing checked
   before dropping it, and unbiasedness is not the property that matters -
   deleting a mean-zero term is not Rao-Blackwellisation and does not inherit
   its variance guarantee.

4. Whether the gap depends on where `q` is standing: not much, and not
   monotonically. Both variances move by more than a factor of 20 between the
   optimum and four standard deviations out, and the ratio between them wanders
   over 1.9x to 10.6x with no trend in the middle. I had predicted the gap would
   widen near the optimum, on the reasoning that the pathwise integrand vanishes
   there while the score function's does not. Both shrink.

5. The ELBO trace cannot see any of it. The two runs finish 6.3e-03 apart on an
   ELBO of 2.4757 and both traces look converged. What separates them is the
   step-to-step jitter, 1.0e-03 against 2.1e-04, which is the estimator's
   standard deviation and is not a quantity the objective reports. Day 1's
   sentence one level up: the ELBO is an expectation under `q`, and here the
   thing it cannot report on is its own sampling noise.

NumPy only. Fixed seeds. About 2 seconds - everything here is batched, which is
why the variance study could afford 400 replicates instead of an error bar.
"""

from typing import Callable, Tuple

import numpy as np

RNG = np.random.default_rng(2)


# ----------------------------------------------------------------------------
# The target, and the exact gradient the estimators are scored against
# ----------------------------------------------------------------------------


def ar1(d: int, r: float) -> np.ndarray:
    """Unit-variance covariance with corr(i, j) = r ** |i - j|. Same as day 1."""
    idx = np.arange(d)
    return r ** np.abs(idx[:, None] - idx[None, :])


def exact_elbo_and_grad(
    q_mean: np.ndarray, log_sd: np.ndarray, mean: np.ndarray, prec: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """ELBO and its exact gradients for diagonal Gaussian q against Gaussian p.

    `ELBO = -0.5 (delta' Lambda delta + tr(Lambda diag(s^2))) + sum log s + c`,
    so `d/dm = -Lambda delta` and `d/dlog s = -diag(Lambda) s^2 + 1`. Written out
    because the point of the file is that the sampled estimators are compared
    against a number rather than against each other.
    """
    sd = np.exp(log_sd)
    delta = q_mean - mean
    d = len(q_mean)
    elbo = (
        -0.5 * (delta @ prec @ delta + float(np.sum(np.diag(prec) * sd**2)))
        + float(np.sum(log_sd))
        + 0.5 * d * (1.0 + np.log(2.0 * np.pi))
    )
    grad_mean = -prec @ delta
    grad_log_sd = -np.diag(prec) * sd**2 + 1.0
    return elbo, grad_mean, grad_log_sd


def log_p_unnormalised(z: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
    delta = z - mean
    return -0.5 * np.einsum("ij,jk,ik->i", delta, prec, delta)


def log_q(z: np.ndarray, q_mean: np.ndarray, log_sd: np.ndarray) -> np.ndarray:
    sd = np.exp(log_sd)
    return float(-np.sum(log_sd) - 0.5 * len(q_mean) * np.log(2.0 * np.pi)) - 0.5 * np.sum(
        ((z - q_mean) / sd) ** 2, axis=1
    )


# ----------------------------------------------------------------------------
# The two estimators
# ----------------------------------------------------------------------------


def score_function_gradient(
    q_mean, log_sd, mean, prec, n_samples, rng, rao_blackwell=False, control_variate=False
):
    """REINFORCE / likelihood-ratio estimator of grad ELBO.

    `grad E_q[f] = E_q[f * grad log q]` with `f = log p~ - log q`. It needs
    nothing from `p` but the ability to evaluate it, which is the selling point,
    and it pays for that in variance because `grad log q` carries the whole
    gradient signal and the sample of `f` multiplies it blind.

    `rao_blackwell` drops the `log q_j`, `j != i`, terms from coordinate i's
    estimator: `q` factorises, so those are independent of `grad_i log q` and
    the product has mean zero. That is the whole of what the mean-field recipe
    has to drop on a target whose `log p` does not factor - and it is named
    after a theorem it is not an instance of. Measured in `main`, it costs 3.7x.

    `control_variate` subtracts `a * grad log q` with the scalar `a` estimated
    from the batch, which is unbiased because `E_q[grad log q] = 0`.
    """
    sd = np.exp(log_sd)
    d = len(q_mean)
    eps = rng.standard_normal((n_samples, d))
    z = q_mean + sd * eps

    log_p = log_p_unnormalised(z, mean, prec)
    f = log_p - log_q(z, q_mean, log_sd)
    # grad of log q wrt (mean, log sd), per sample
    g_mean = eps / sd
    g_log_sd = eps**2 - 1.0

    if rao_blackwell:
        # q factorises, so for j != i the factor log q_j depends only on z_j and
        # is independent of grad_i log q, whose mean is zero - the product's
        # expectation is therefore zero and the term is pure noise. drop it and
        # keep log q_i, which does depend on z_i. log p does not factor on a
        # correlated target, so this is the only thing there is to drop.
        log_q_i = -(log_sd + 0.5 * np.log(2 * np.pi)) - 0.5 * eps**2
        f_i = log_p[:, None] - log_q_i
    else:
        f_i = np.repeat(f[:, None], d, axis=1)

    terms_mean = f_i * g_mean
    terms_log_sd = f_i * g_log_sd
    if control_variate:
        terms_mean = _cv(terms_mean, g_mean)
        terms_log_sd = _cv(terms_log_sd, g_log_sd)
    return terms_mean.mean(axis=0), terms_log_sd.mean(axis=0)


def _cv(terms: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Subtract a * control per coordinate, a = Cov(terms, control)/Var(control)."""
    t_c = terms - terms.mean(axis=0)
    c_c = control - control.mean(axis=0)
    var = (c_c**2).mean(axis=0)
    a = np.where(var > 0, (t_c * c_c).mean(axis=0) / np.where(var > 0, var, 1.0), 0.0)
    return terms - a * control


def reparameterised_gradient(q_mean, log_sd, mean, prec, n_samples, rng):
    """Pathwise estimator: differentiate through `z = m + s * eps`.

    `grad_m = E[grad_z log p~(z)]` and `grad_logs = E[grad_z log p~(z) * s * eps]
    + 1`, the `+1` being the entropy's exact derivative rather than a sampled
    one. It uses the gradient of `log p`, which the score-function estimator does
    not need, and that is exactly the extra information the variance buys.
    """
    sd = np.exp(log_sd)
    d = len(q_mean)
    eps = rng.standard_normal((n_samples, d))
    z = q_mean + sd * eps
    grad_z = -(z - mean) @ prec
    return grad_z.mean(axis=0), (grad_z * sd * eps).mean(axis=0) + 1.0


# ----------------------------------------------------------------------------


def batch_statistics(
    estimator: Callable[[], Tuple[np.ndarray, np.ndarray]], n_batches: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and variance across `n_batches` independent estimates."""
    means, log_sds = [], []
    for _ in range(n_batches):
        g_mean, g_log_sd = estimator()
        means.append(g_mean)
        log_sds.append(g_log_sd)
    means = np.array(means)
    log_sds = np.array(log_sds)
    return (
        np.array([means.mean(axis=0), means.var(axis=0, ddof=1)]),
        np.array([log_sds.mean(axis=0), log_sds.var(axis=0, ddof=1)]),
    )


def ascend(q_mean, log_sd, mean, prec, grad_fn, steps, lr):
    """Plain gradient ascent on the ELBO with whichever estimator is passed in."""
    q_mean = q_mean.copy()
    log_sd = log_sd.copy()
    trace = []
    for _ in range(steps):
        g_mean, g_log_sd = grad_fn(q_mean, log_sd)
        q_mean = q_mean + lr * g_mean
        log_sd = log_sd + lr * g_log_sd
        trace.append(exact_elbo_and_grad(q_mean, log_sd, mean, prec)[0])
    return q_mean, log_sd, np.array(trace)


def main():
    np.set_printoptions(precision=4, suppress=True)

    d = 4
    cov = ar1(d, 0.6)
    prec = np.linalg.inv(cov)
    mean = np.array([0.5, -1.0, 2.0, 0.0])

    # the mean-field optimum from day 1: mean exact, variance the conditional one
    opt_mean = mean.copy()
    opt_log_sd = 0.5 * np.log(1.0 / np.diag(prec))

    print("== both estimators are unbiased for the exact gradient ==")
    print("   d = 4, AR(1) r = 0.6, q one standard deviation off the optimum")
    q_mean = opt_mean + 1.0
    log_sd = opt_log_sd + 0.3
    _, true_mean_grad, true_log_sd_grad = exact_elbo_and_grad(q_mean, log_sd, mean, prec)

    n_batches, n_samples = 400, 512
    rng = np.random.default_rng(11)
    sf = batch_statistics(
        lambda: score_function_gradient(q_mean, log_sd, mean, prec, n_samples, rng),
        n_batches,
    )
    rng = np.random.default_rng(11)
    rp = batch_statistics(
        lambda: reparameterised_gradient(q_mean, log_sd, mean, prec, n_samples, rng),
        n_batches,
    )
    for name, stats, truth in (
        ("d/dmean   ", (sf[0], rp[0]), true_mean_grad),
        ("d/dlog sd ", (sf[1], rp[1]), true_log_sd_grad),
    ):
        sf_stat, rp_stat = stats
        sf_se = np.sqrt(sf_stat[1] / n_batches)
        rp_se = np.sqrt(rp_stat[1] / n_batches)
        print(f"  {name} exact            {truth}")
        print(f"             score-function   {sf_stat[0]}  ({np.abs(sf_stat[0] - truth).max() / sf_se.max():.4f} se)")
        print(f"             reparameterised  {rp_stat[0]}  ({np.abs(rp_stat[0] - truth).max() / rp_se.max():.4f} se)")

    print("\n== so the difference is variance, per batch of 512 ==")
    print("  coordinate            score-function   reparameterised      ratio")
    for name, sf_stat, rp_stat in (
        ("mean, coord 0     ", sf[0], rp[0]),
        ("log sd, coord 0   ", sf[1], rp[1]),
    ):
        print(
            f"  {name}  {sf_stat[1][0]:14.6e}   {rp_stat[1][0]:14.6e}"
            f"   {sf_stat[1][0] / rp_stat[1][0]:9.1f}x"
        )

    print("\n== the standard variance reduction recipe, on this target ==")
    for label, rb, cv in (
        ("plain                 ", False, False),
        ("control variate only  ", False, True),
        ("term dropped          ", True, False),
        ("dropped + control var ", True, True),
    ):
        rng = np.random.default_rng(11)
        stats_mean, _ = batch_statistics(
            lambda: score_function_gradient(
                q_mean, log_sd, mean, prec, n_samples, rng, rao_blackwell=rb, control_variate=cv
            ),
            n_batches,
        )
        bias = np.abs(stats_mean[0] - true_mean_grad).max()
        print(
            f"  {label} var(d/dmean, coord 0) = {stats_mean[1][0]:.6e}"
            f"   |bias|max = {bias:.4f}"
        )

    print("\n== why dropping it costs rather than saves, in one covariance ==")
    print("  full = A - B, with B the sum_{j != i} log q_j terms that go")
    rng = np.random.default_rng(3)
    sd = np.exp(log_sd)
    eps = rng.standard_normal((400_000, d))
    z = q_mean + sd * eps
    lp = log_p_unnormalised(z, mean, prec)
    lq = log_q(z, q_mean, log_sd)
    lq_i = -(log_sd + 0.5 * np.log(2 * np.pi)) - 0.5 * eps**2
    g = eps / sd
    a_term = (lp[:, None] - lq_i) * g
    b_term = (lq[:, None] - lq_i) * g
    full = a_term - b_term
    cov_ab = float(np.cov(a_term[:, 0], b_term[:, 0])[0, 1])
    var_b = float(b_term[:, 0].var())
    print(f"  E[B]        {b_term[:, 0].mean():+.5f}   (zero, so dropping B is unbiased)")
    print(f"  Var(full)   {full[:, 0].var():8.4f}")
    print(f"  Var(A)      {a_term[:, 0].var():8.4f}")
    print(f"  Var(B)      {var_b:8.4f}   Cov(A, B) {cov_ab:8.4f}")
    print(f"  2Cov - Var(B) = {2 * cov_ab - var_b:+8.4f}, which is what dropping B adds")
    print(f"  corr(A, B)  {cov_ab / np.sqrt(a_term[:, 0].var() * var_b):8.4f}")

    print("\n== does the gap depend on where q is standing? ==")
    print("  offset   score-function          reparameterised          ratio")
    for offset in (0.0, 0.5, 1.0, 2.0, 4.0):
        qm = opt_mean + offset
        ls = opt_log_sd + 0.3 * offset
        rng = np.random.default_rng(5)
        sf_stat, _ = batch_statistics(
            lambda: score_function_gradient(qm, ls, mean, prec, n_samples, rng), 200
        )
        rng = np.random.default_rng(5)
        rp_stat, _ = batch_statistics(
            lambda: reparameterised_gradient(qm, ls, mean, prec, n_samples, rng), 200
        )
        print(
            f"  {offset:5.1f}   {sf_stat[1][0]:14.6e}   {rp_stat[1][0]:14.6e}"
            f"   {sf_stat[1][0] / rp_stat[1][0]:9.1f}x"
        )

    print("\n== and the same variance gap as dimension grows ==")
    print("  d      ratio at the optimum")
    for dim in (2, 4, 8, 16):
        cov_d = ar1(dim, 0.6)
        prec_d = np.linalg.inv(cov_d)
        mean_d = np.zeros(dim)
        qm = mean_d + 1.0
        ls = 0.5 * np.log(1.0 / np.diag(prec_d))
        rng = np.random.default_rng(3)
        sf_stat, _ = batch_statistics(
            lambda: score_function_gradient(qm, ls, mean_d, prec_d, n_samples, rng), 200
        )
        rng = np.random.default_rng(3)
        rp_stat, _ = batch_statistics(
            lambda: reparameterised_gradient(qm, ls, mean_d, prec_d, n_samples, rng), 200
        )
        print(f"  {dim:3d}    {sf_stat[1][0] / rp_stat[1][0]:9.1f}x")

    print("\n== what the ELBO trace reports about all of this ==")
    start_mean = mean + 2.0
    start_log_sd = np.zeros(d)
    rng = np.random.default_rng(21)
    _, _, trace_sf = ascend(
        start_mean,
        start_log_sd,
        mean,
        prec,
        lambda m, s: score_function_gradient(m, s, mean, prec, 64, rng),
        steps=800,
        lr=0.01,
    )
    rng = np.random.default_rng(21)
    _, _, trace_rp = ascend(
        start_mean,
        start_log_sd,
        mean,
        prec,
        lambda m, s: reparameterised_gradient(m, s, mean, prec, 64, rng),
        steps=800,
        lr=0.01,
    )
    best = exact_elbo_and_grad(opt_mean, opt_log_sd, mean, prec)[0]
    jitter_sf = float(np.std(np.diff(trace_sf[-200:])))
    jitter_rp = float(np.std(np.diff(trace_rp[-200:])))
    print(f"  optimal ELBO                {best:.6f}")
    print(f"  score-function, final       {trace_sf[-1]:.6f}   gap {best - trace_sf[-1]:.3e}")
    print(f"  reparameterised, final      {trace_rp[-1]:.6f}   gap {best - trace_rp[-1]:.3e}")
    print(f"  final ELBOs differ by       {abs(trace_sf[-1] - trace_rp[-1]):.3e}")
    print(f"  step-to-step sd, last 200   {jitter_sf:.3e} vs {jitter_rp:.3e}")

    # the file's point, asserted rather than described. the two estimators are
    # the same in the mean and differ in variance by a factor that is set by
    # the dimension rather than by the estimators, and the objective they are
    # both climbing reports the first and not the second. day 1's sentence one
    # level up: the ELBO is an expectation under q and cannot report on its own
    # sampling noise.
    assert sf[0][1][0] > rp[0][1][0], "reparameterisation should win on variance"
    assert abs(trace_sf[-1] - trace_rp[-1]) < 1e-2, "the traces should be hard to tell apart"
    assert 2 * cov_ab - var_b > 0, "the dropped term was doing work as a control variate"


if __name__ == "__main__":
    main()
