# Gaussian Process Regression from Scratch

Exact Gaussian process regression written from scratch in NumPy. Kernels, the
posterior via a Cholesky solve, hyperparameters learned by maximizing the log
marginal likelihood, and predictive uncertainty that is actually checked for
calibration rather than assumed.

## Idea

A GP is a distribution over functions: for any finite set of inputs `X`, the
function values `f(X)` are jointly Gaussian with covariance `K(X, X)` given by a
kernel. Regression is then just conditioning a Gaussian on the observed half,
which has a closed form:

    mean = K*^T (K + s^2 I)^-1 y
    cov  = K** - K*^T (K + s^2 I)^-1 K*

Everything else in the project is either making that numerically safe or
deciding what the kernel's parameters should be.

The kernel is the prior. RBF gives infinitely differentiable sample paths, Matern
3/2 gives paths that are once differentiable and visibly rougher, periodic
encodes exact repetition. Choosing between them is a modelling statement, and the
log marginal likelihood turns out to be enough to make that choice on the
training set alone.

## Layout

- `day1_kernels.py` - RBF, Matern 3/2 and periodic kernels, the Gram matrix,
  positive-definiteness checks and prior sample paths. Includes the
  demonstration that the RBF Gram matrix of 120 dense points is numerically
  singular without jitter.
- `day2_posterior.py` - the exact posterior via Cholesky. Predictive mean and
  full covariance, joint posterior sample paths, and a comparison against the
  textbook `inv(K)` formulation to show how much accuracy the explicit inverse
  throws away.
- `day3_marginal_likelihood.py` - the log marginal likelihood, its analytic
  gradients in log space, a central-difference gradient check, and multi-restart
  L-BFGS-B hyperparameter fitting. The lengthscale sweep is the Occam's-razor
  argument in table form.
- `day4_uncertainty.py` - calibration (coverage, band width, NLPD), a fair
  kernel comparison with each kernel fit at its own optimum, and the ridge
  baselines.
- `test_posterior.py`, `test_marginal_likelihood.py` - the properties that must
  hold: posterior variance collapsing at training points, agreement with the
  naive formulation, gradient correctness.

## Key design choices

- **Cholesky, never `inv`.** The posterior mean is a `cho_solve`, and the
  covariance uses only the forward triangular solve so the subtracted term is
  manifestly PSD. Forming `K^-1` explicitly costs accuracy for no speed.
- **Hyperparameters optimized in log space.** Lengthscale, signal variance and
  noise variance are all strictly positive; optimizing their logs removes the
  constraints and makes the steps multiplicative, which is what you want when
  the right scale is unknown to an order of magnitude.
- **One factorization for the objective and all gradients.** The identity
  `d/dtheta log p = 0.5 tr((alpha alpha^T - K_y^-1) dK/dtheta)` shares a single
  Cholesky, and the trace is computed as `sum(A * B.T)` in O(n^2) rather than
  through an O(n^3) matrix product.
- **Jitter is separate from noise, conceptually.** They occupy the same diagonal
  slot and the arithmetic cannot tell them apart, but one is a modelling
  statement and the other is an admission about floating point.

## Results

Trained on 40 noisy points from `sin(3x) + 0.3x` with noise sd 0.15, evaluated on
200 held-out points over the same range.

| Kernel | Lengthscale | Learned noise var | LML | Held-out RMSE | 95% coverage |
|---|---|---|---|---|---|
| RBF | 0.614 | 0.0288 | -10.32 | 0.171 | 97.5% |
| Matern 3/2 | 0.896 | 0.0214 | -12.70 | 0.180 | 96.5% |
| Periodic | 2.576 | 0.3000 | -39.85 | 0.528 | 99.0% |

The LML ranking and the RMSE ranking agree, which is the point of being able to
compare kernels without a validation split. Periodic is given the correct period
and still loses badly, because it cannot represent the linear drift - the case
for composite kernels in one row.

Against the baselines, on the same split:

| Model | Held-out RMSE | Extrapolation RMSE (\|x\|>3) |
|---|---|---|
| GP / kernel ridge (RBF) | 0.171 | 0.39 |
| Polynomial ridge, degree 3 | 0.673 | 2.43 |
| Polynomial ridge, degree 5 | 0.662 | 4.19 |
| Polynomial ridge, degree 9 | 0.187 | 97.97 |

## The thing worth taking away

Kernel ridge regression with the same kernel and `lambda = s^2` produces the GP
posterior mean to machine precision - `day4_uncertainty.py` asserts the two agree
to within 1e-14. They are the same estimator derived twice, once as penalized
least squares in an RKHS and once by conditioning a Gaussian.

So the GP buys nothing in the first moment. What it buys is everything else: a
predictive variance, a marginal likelihood that selects hyperparameters and
kernels without held-out data, and calibrated intervals. On the held-out split
the 95% bands cover 97.5% - slightly conservative, traceable to a learned noise
variance of 0.0288 against a true 0.0225. Out at `x = 4`, two lengthscales past
the data, the band widens roughly fivefold while the error only doubles, so the
prediction stays inside its interval. It is a correct statement and a useless
prediction at once, which is the failure mode you want: the model reports its own
ignorance instead of extrapolating confidently, and the degree-9 polynomial
sitting at RMSE 98 with nothing to say about it is the alternative.

## Running

```bash
python day1_kernels.py
python day2_posterior.py
python day3_marginal_likelihood.py
python day4_uncertainty.py
pytest test_posterior.py test_marginal_likelihood.py
```

Requires NumPy and SciPy.
