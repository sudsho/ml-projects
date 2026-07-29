"""
Day 1 of Gaussian process regression from scratch.

A Gaussian process is a distribution over functions: for any finite set of input
points X, the vector of function values f(X) is jointly Gaussian with mean m(X)
and covariance K(X, X). Everything interesting about a GP therefore lives in the
kernel k(x, x'), which says how strongly the function values at two inputs are
correlated. Choosing a kernel *is* choosing the prior over functions - smooth,
rough, periodic, slowly varying - and the rest of the method is mechanical.

This file builds the kernels themselves plus the machinery to interrogate them:

  - the squared-exponential (RBF) kernel, whose samples are infinitely
    differentiable and so produce very smooth function draws;
  - the Matern 3/2 kernel, which is only once differentiable and gives visibly
    rougher paths - usually the more honest prior for physical data;
  - a periodic kernel, which encodes exact repetition at a known period by
    warping the input onto a circle before measuring distance.

Every kernel takes a lengthscale (how far apart inputs must be before their
values decorrelate) and an output variance (the marginal prior variance of the
function). Both are strictly positive, so later on they will be optimized in log
space; here they are just parameters.

The other half of the file is the Gram matrix and its sanity checks. K(X, X)
must be symmetric positive semi-definite to be a valid covariance, and the
practical test is whether a Cholesky factorization succeeds. In floating point
the RBF Gram matrix of closely spaced points is famously ill-conditioned, so a
small "jitter" is added to the diagonal - which is exactly the same slot that
observation noise will occupy tomorrow, a coincidence worth noticing early.
"""

import numpy as np


def pairwise_sq_dists(xa, xb):
    """Squared Euclidean distances between every row of xa and every row of xb.

    Uses the expansion ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 so the whole
    matrix comes from one matmul instead of a Python loop. The expansion can go
    slightly negative from cancellation when a and b are nearly equal, so the
    result is clipped at zero before it is ever handed to a sqrt.
    """
    xa = np.atleast_2d(xa)
    xb = np.atleast_2d(xb)
    sq_a = np.sum(xa ** 2, axis=1)[:, None]
    sq_b = np.sum(xb ** 2, axis=1)[None, :]
    d2 = sq_a + sq_b - 2.0 * (xa @ xb.T)
    return np.maximum(d2, 0.0)


def rbf_kernel(xa, xb, lengthscale=1.0, variance=1.0):
    """Squared-exponential kernel: variance * exp(-r^2 / (2 l^2)).

    The smoothest common choice - sample paths are infinitely differentiable,
    which is why RBF priors often look implausibly clean next to real data.
    Correlation decays like a Gaussian in the distance, so points more than a
    few lengthscales apart are effectively independent.
    """
    d2 = pairwise_sq_dists(xa, xb)
    return variance * np.exp(-0.5 * d2 / lengthscale ** 2)


def matern32_kernel(xa, xb, lengthscale=1.0, variance=1.0):
    """Matern 3/2: variance * (1 + sqrt(3) r / l) * exp(-sqrt(3) r / l).

    Sample paths are once differentiable and no more, so draws look rougher than
    RBF at the same lengthscale. The polynomial prefactor is what makes the tail
    decay exponentially in r rather than r^2, so the kernel is far less
    aggressive about forcing distant points to agree.
    """
    r = np.sqrt(pairwise_sq_dists(xa, xb))
    scaled = np.sqrt(3.0) * r / lengthscale
    return variance * (1.0 + scaled) * np.exp(-scaled)


def periodic_kernel(xa, xb, lengthscale=1.0, variance=1.0, period=1.0):
    """MacKay's periodic kernel: variance * exp(-2 sin^2(pi r / p) / l^2).

    Wrapping the distance through a sine makes inputs exactly one period apart
    perfectly correlated, so the prior only contains functions that repeat. Note
    this is a 1-D construction - it uses the raw coordinate difference, not the
    Euclidean norm, so it is applied per-dimension in the multi-D case.
    """
    xa = np.atleast_2d(xa)
    xb = np.atleast_2d(xb)
    if xa.shape[1] != 1 or xb.shape[1] != 1:
        raise ValueError("periodic_kernel here is 1-D only")
    r = np.abs(xa - xb.T)
    sin_term = np.sin(np.pi * r / period)
    return variance * np.exp(-2.0 * sin_term ** 2 / lengthscale ** 2)


def gram_matrix(kernel, x, jitter=1e-8, **kernel_kwargs):
    """Build K(x, x) and add jitter to the diagonal for numerical stability.

    The jitter is not a modelling choice, it is a floating-point one: the RBF
    Gram matrix of densely sampled inputs has eigenvalues that decay to below
    machine precision, and Cholesky then fails on a matrix that is positive
    definite in exact arithmetic. Tomorrow the observation-noise variance lands
    in this same diagonal slot, which is why noisy GP regression is better
    conditioned than the noise-free case.
    """
    k = kernel(x, x, **kernel_kwargs)
    # symmetrize explicitly - the matmul in pairwise_sq_dists is not exactly
    # symmetric in floating point, and Cholesky only reads one triangle anyway
    # so an asymmetry would silently pass unnoticed.
    k = 0.5 * (k + k.T)
    return k + jitter * np.eye(k.shape[0])


def is_positive_definite(k):
    """Cholesky-based PD test - cheaper and stricter than eigendecomposition."""
    try:
        np.linalg.cholesky(k)
        return True
    except np.linalg.LinAlgError:
        return False


def sample_prior(kernel, x, n_samples=5, jitter=1e-8, rng=None, **kernel_kwargs):
    """Draw function values from the zero-mean GP prior at inputs x.

    The standard trick: if K = L L^T then L @ u with u ~ N(0, I) has covariance
    L L^T = K exactly, so one Cholesky gives arbitrarily many correlated draws
    without ever forming a matrix square root.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    k = gram_matrix(kernel, x, jitter=jitter, **kernel_kwargs)
    chol = np.linalg.cholesky(k)
    u = rng.standard_normal(size=(k.shape[0], n_samples))
    return chol @ u


def condition_number(k):
    """Ratio of largest to smallest eigenvalue - how close K is to singular."""
    eigvals = np.linalg.eigvalsh(k)
    return eigvals[-1] / max(eigvals[0], 1e-300)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(-3.0, 3.0, 120)[:, None]

    kernels = {
        "rbf": (rbf_kernel, {"lengthscale": 1.0, "variance": 1.0}),
        "matern32": (matern32_kernel, {"lengthscale": 1.0, "variance": 1.0}),
        "periodic": (periodic_kernel, {"lengthscale": 1.0, "variance": 1.0, "period": 2.0}),
    }

    for name, (fn, kwargs) in kernels.items():
        k = gram_matrix(fn, x, jitter=1e-8, **kwargs)
        print(f"{name:9s} shape={k.shape} diag={k[0, 0]:.4f} "
              f"pd={is_positive_definite(k)} cond={condition_number(k):.3e}")

    # the jitter is doing real work here - drop it and RBF on 120 dense points
    # is numerically singular even though it is PD in exact arithmetic.
    raw = rbf_kernel(x, x, lengthscale=1.0, variance=1.0)
    print(f"\nrbf without jitter: pd={is_positive_definite(raw)} "
          f"cond={condition_number(raw):.3e}")

    # lengthscale controls how fast correlation dies off, which is the single
    # most consequential knob in the whole model.
    print("\nrbf correlation at distance 1.0 by lengthscale:")
    for ls in (0.25, 0.5, 1.0, 2.0):
        corr = rbf_kernel(np.zeros((1, 1)), np.ones((1, 1)), lengthscale=ls)[0, 0]
        print(f"  l={ls:<5} k(0, 1)={corr:.6f}")

    draws = sample_prior(rbf_kernel, x, n_samples=5, rng=rng, lengthscale=1.0)
    print(f"\nprior draws {draws.shape}, empirical sd={draws.std():.3f} (prior sd=1.0)")
