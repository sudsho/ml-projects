"""Day 4 of epistemic uncertainty: scoring the methods with proper scoring rules.

Three days of ranking these methods by one diagnostic - the ratio of the reported
epistemic standard deviation inside the gap to the one outside it, against a
control trained on data with the gap filled in. That diagnostic is mine. Nobody
optimizes it, it has no decision-theoretic standing, and day 3 ended with a
ranking built entirely out of it: Laplace at a ratio of 19.4, the ensemble at
10.3, MC dropout at nothing.

Today the same four predictive distributions are scored on held-out data by two
proper scoring rules, which do have standing, and the answer is that **they
disagree with each other almost completely.**

    in-gap ranking by NLL : laplace  < dropout < ensemble < single
    in-gap ranking by CRPS: ensemble < dropout < single   < laplace

Laplace is first on one and last on the other. Not close on either - it wins NLL
by 0.49 nats over the next method and loses CRPS to the ensemble by 54%.

Both are proper. Neither is a proxy for the other, and I had been using "proper
scoring rule" as though it named one thing.

**Why they split.** NLL is `-log p(y)` and is unbounded below, so it is dominated
by the points where a method put almost no mass on what happened. The single head
scores 13.273 in the gap with a 99th percentile of 56.07 - that tail is the whole
number, and it is what "confidently wrong" costs when the cost is measured in
nats. CRPS is an integrated squared CDF error, lives in the units of `y`, and is
bounded by roughly the size of the miss. A distribution that puts 1e-20 on the
truth and one that puts 1e-3 on it are 40 nats apart and nearly the same CRPS.

So NLL asks whether the method admitted the outcome was possible, and CRPS asks
how far off it was in the target's own units. In a region with no training data
those are different questions, and the methods sort differently on them.

**The mean is doing most of the CRPS ordering.** Mean absolute error in the gap:
ensemble 0.4365, single 0.5443, dropout 0.5826, Laplace 0.6730. CRPS in the gap:
ensemble 0.3327, dropout 0.4390, single 0.4779, Laplace 0.5139. Same order except
that dropout and the single head swap - dropout has the worse mean and the better
CRPS, because it has some spread and the single head has essentially none.

That is CRPS working as advertised: it rewards calibrated spread, and it rewards
it much less than it rewards a good mean. The ensemble wins on CRPS largely
because averaging ten independently initialized means is variance reduction on the
mean itself, which is a real benefit of ensembling that has nothing to do with
uncertainty quantification at all. Three days of treating the ensemble as an
epistemic-uncertainty method and its best score here is earned by its point
prediction.

**Calibration goes with NLL and it is not close.** Mean absolute coverage gap in
the gap region: Laplace 0.062, ensemble 0.287, dropout 0.347, single 0.417.
Laplace's central intervals in the gap land at 0.50 nominal / 0.50 empirical and
0.90 / 0.81. The ensemble's 90% interval covers 57% and its 50% covers 14%. The
single head's 90% interval covers 19% of the points it was asked about.

So the only method that is honest in the gap is also the least accurate in it, and
it is last by the score that measures accuracy. That sentence is the day.

**What the diagnostic was tracking.** Day 3's epistemic ratio put Laplace first,
the ensemble second, dropout nowhere. That is the NLL ordering and the calibration
ordering, and it is the reverse of the CRPS ordering on its top and bottom
entries. The diagnostic was never method-neutral - it measures whether the
reported spread grows where the data stops, which is exactly what NLL rewards and
what CRPS mostly ignores. Three days of it looked like a general-purpose ranking
and it was one of the two available rankings, chosen before I knew there were two.

**On the data there is nothing to choose between them.** NLL 0.339 to 0.387, CRPS
0.198 to 0.201, calibration error under 0.05 for all four. Every difference above
is a difference in the gap, which is day 1's design working: the methods are
separated only where the training data is absent, so the comparison is about
epistemic uncertainty rather than about fit.

**One confound I cannot remove today.** The single head trains without weight
decay and the Laplace model trains with it, because the Gauss-Newton Hessian needs
a prior precision to be a posterior at all. So the mean-error difference between
those two - 0.5443 against 0.6730 - is a regularization difference and not
anything Laplace did. Laplace does not touch the mean; it adds an epistemic term
to a fit that already exists. The comparison that is clean is Laplace against the
dropout model at `rate=0.0`, which is the same fit, and that one is a comparison
of variances only.

The conclusion is not a winner. It is that "best at epistemic uncertainty" was not
a well-posed question, and which method to reach for depends on what the
uncertainty is for:

- if a downstream step thresholds on probability - abstention, gating, anything
  that asks "could this have happened?" - the honest density is what matters, and
  last-layer Laplace gets it for one training run and a 65x65 solve.
- if the loss is in the units of the target, the ensemble's mean is worth its ten
  runs and its badly calibrated intervals are affordable.
- MC dropout is second on both scores and detects nothing against its own control
  (day 3), which is the combination that would have got it adopted: it looks
  reasonable by every number here and the number that tests it is the one nobody
  computes.
"""

import time

import numpy as np
import torch

from day1_the_gap import (
    DOMAIN,
    GAP,
    predict,
    sample_dataset,
    train_gaussian_head,
    true_mean,
)
from day2_deep_ensembles import ENSEMBLE_SIZE, region_masks, train_ensemble
from day3_dropout_and_laplace import (
    DROPOUT_PASSES,
    last_layer_laplace,
    laplace_predict,
    mc_dropout_samples,
    train_dropout_head,
)

CALIBRATION_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# how many mixture components the CRPS double sum is allowed to see. the ensemble
# has 10 and the single-Gaussian methods have 1, so this only ever bites on MC
# dropout's 200 passes, where the exact pairwise term is a (200, 200) block per
# test point. a uniform thinning of MC samples is still a sample from the same
# approximate posterior, so this changes the Monte Carlo error and not the object
# being scored - unlike capping the ensemble, which would change the method.
CRPS_MAX_COMPONENTS = 50


def _normal_cdf(z):
    """Standard normal CDF, via `torch.erf` so this file needs no scipy.

    Ravelled and reshaped rather than passed straight through, because
    `ascontiguousarray` promotes a scalar to shape `(1,)` and the callers below
    are given scalars in the checks and arrays everywhere else. Preserving the
    input shape here is cheaper than every caller knowing about it.
    """
    array = np.asarray(z, dtype=np.float64)
    flat = torch.from_numpy(np.ascontiguousarray(array.ravel()))

    return (0.5 * (1.0 + torch.erf(flat / np.sqrt(2.0)))).numpy().reshape(array.shape)


def _normal_pdf(z):
    return np.exp(-0.5 * np.asarray(z, dtype=np.float64) ** 2) / np.sqrt(2.0 * np.pi)


def _folded_normal_mean(z):
    """`E|X|` for `X ~ N(z, 1)`, which is `2 phi(z) + z (2 Phi(z) - 1)`.

    The one special function every closed-form CRPS below is built out of. Both
    the score of a Gaussian against a point and the pairwise term of a Gaussian
    mixture are this, rescaled.
    """
    return 2.0 * _normal_pdf(z) + z * (2.0 * _normal_cdf(z) - 1.0)


def gaussian_crps(mu, sigma, y):
    """CRPS of `N(mu, sigma^2)` against observations `y`, in closed form.

    `sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]` with `z = (y - mu)/sigma`,
    which is `sigma * (A(z) - 1/sqrt(pi))` for the folded-normal mean `A` above -
    the `1/sqrt(pi)` being `E|X - X'|` for two independent standard normals.

    In the same units as `y`, unlike NLL, which is the reason both are here.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    z = (np.asarray(y, dtype=np.float64) - mu) / sigma

    return sigma * (_folded_normal_mean(z) - 1.0 / np.sqrt(np.pi))


def mixture_crps(mus, sigmas, y, chunk=64):
    """CRPS of a uniform Gaussian mixture against `y`, in closed form.

    `mus` and `sigmas` are `(T, n)` - `T` components at each of `n` points.

        CRPS = (1/T) sum_i E|X_i - y| - (1/(2 T^2)) sum_i sum_j E|X_i - X_j|

    with `X_i ~ N(mu_i, sigma_i^2)`. Both expectations are folded normals, so the
    first term is `sigma_i A((y - mu_i)/sigma_i)` and the pairwise one is
    `s_ij A((mu_i - mu_j)/s_ij)` with `s_ij = sqrt(sigma_i^2 + sigma_j^2)`.

    This is the mixture's own CRPS and not the CRPS of the Gaussian that shares
    its mean and variance. Scoring the Gaussian summary instead would be day 2's
    mistake in a new place: the summary is for reporting and the mixture is what
    the method actually predicts, and they come apart exactly where the components
    disagree, which is the region under study.

    Chunked over points because the pairwise term is `(T, T)` per point, and MC
    dropout arrives with `T = 200`.
    """
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if mus.shape[0] > CRPS_MAX_COMPONENTS:
        step = mus.shape[0] // CRPS_MAX_COMPONENTS
        mus = mus[::step][:CRPS_MAX_COMPONENTS]
        sigmas = sigmas[::step][:CRPS_MAX_COMPONENTS]

    components = mus.shape[0]
    out = np.empty(mus.shape[1], dtype=np.float64)

    for start in range(0, mus.shape[1], chunk):
        stop = min(start + chunk, mus.shape[1])
        mu = mus[:, start:stop]
        sigma = sigmas[:, start:stop]
        target = y[start:stop]

        first = (sigma * _folded_normal_mean((target[None, :] - mu) / sigma)).mean(axis=0)

        pair_scale = np.sqrt(sigma[:, None, :] ** 2 + sigma[None, :, :] ** 2)
        pair = pair_scale * _folded_normal_mean(
            (mu[:, None, :] - mu[None, :, :]) / pair_scale
        )
        out[start:stop] = first - 0.5 * pair.sum(axis=(0, 1)) / components ** 2

    return out


def mixture_nll(mus, sigmas, y):
    """Negative log density of `y` under the uniform mixture, by log-sum-exp.

    Same quantity day 2's `mixture_log_prob` computes from a list of models,
    negated and taken from component arrays instead, so that all four methods -
    two of which have no model list to hand - go through one code path. The
    `__main__` block checks the two agree on the ensemble.
    """
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    logs = (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigmas)
        - 0.5 * ((y[None, :] - mus) / sigmas) ** 2
    )
    top = logs.max(axis=0)

    return -(top + np.log(np.exp(logs - top).mean(axis=0)))


def mixture_pit(mus, sigmas, y):
    """`F(y)` under the mixture - the probability integral transform.

    Uniform on `[0, 1]` if and only if the predictive distribution is calibrated,
    which is what makes it the right object to build a calibration curve from. It
    also works unchanged for a mixture, where reading off a central interval would
    need the quantile function and the quantile function has no closed form.
    """
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    return _normal_cdf((y[None, :] - mus) / sigmas).mean(axis=0)


def calibration_curve(pit, levels=CALIBRATION_LEVELS):
    """Empirical coverage of the central credible interval at each nominal level.

    `y` falls in the central interval of mass `q` exactly when `|PIT - 0.5| <= q/2`,
    so the whole curve comes off the PIT values with no quantiles anywhere.
    """
    centred = np.abs(np.asarray(pit, dtype=np.float64) - 0.5)

    return np.array([float((centred <= level / 2.0).mean()) for level in levels])


def calibration_error(pit, levels=CALIBRATION_LEVELS):
    """Mean absolute gap between nominal and empirical coverage, over the levels.

    One number for a curve, so the methods fit in a table. Signed information is
    lost, which matters, so the curve is printed too and the sign is read there -
    every method in this project misses in the same direction anyway.
    """
    return float(np.abs(calibration_curve(pit, levels) - levels).mean())


def score(mus, sigmas, x_test, y_test):
    """NLL, CRPS and calibration for one method, split inside and outside the gap.

    Takes component arrays rather than a model, so a single Gaussian (`T = 1`), a
    ten-member ensemble and two hundred dropout passes are all scored by the same
    code with no per-method branch. That is the point of the shape: the three
    methods disagree about what a posterior is and agree completely about what a
    predictive distribution is, and scoring happens on the second thing.
    """
    inside, outside = region_masks(x_test)

    nll = mixture_nll(mus, sigmas, y_test)
    crps = mixture_crps(mus, sigmas, y_test)
    pit = mixture_pit(mus, sigmas, y_test)

    return {
        "nll_gap": float(nll[inside].mean()),
        "nll_data": float(nll[outside].mean()),
        "crps_gap": float(crps[inside].mean()),
        "crps_data": float(crps[outside].mean()),
        "calibration_gap": calibration_error(pit[inside]),
        "calibration_data": calibration_error(pit[outside]),
        "curve_gap": calibration_curve(pit[inside]),
        "curve_data": calibration_curve(pit[outside]),
        "nll_worst_gap": float(np.percentile(nll[inside], 99)),
        "crps_worst_gap": float(np.percentile(crps[inside], 99)),
        "mean_error_gap": float(
            np.abs(mus.mean(axis=0) - true_mean(x_test))[inside].mean()
        ),
    }


def ensemble_components(models, x):
    """`(T, n)` means and sigmas from a list of models."""
    mus = np.stack([predict(m, x)[0] for m in models])
    sigmas = np.stack([predict(m, x)[1] for m in models])

    return mus.astype(np.float64), sigmas.astype(np.float64)


def laplace_components(model, covariance, x):
    """Laplace as a one-component mixture, with the epistemic term folded into it.

    The predictive is exactly Gaussian here and does not need a mixture to
    represent it: the mean is linear in the last-layer weights, a Gaussian pushed
    through a linear map is Gaussian, and adding the aleatoric variance keeps it
    so. So `T = 1` with `sigma^2 = aleatoric + phi^T Sigma phi` is not a summary
    of the predictive, it *is* the predictive - which is why this method can be
    put through the mixture scorer without the concession day 2 refused to make
    for the ensemble.
    """
    mu, aleatoric, epistemic = laplace_predict(model, covariance, x)
    total = aleatoric + np.maximum(epistemic, 0.0)

    return mu[None, :].astype(np.float64), np.sqrt(total)[None, :].astype(np.float64)


if __name__ == "__main__":
    torch.set_num_threads(1)

    x_train, y_train = sample_dataset(600, seed=0)

    # the test set has no hole in it. that is the whole design and it is worth
    # being explicit about, because it is the one thing here that could not be
    # done on a real dataset: the gap is a hole i cut, so i can sample the
    # generative process inside it and ask what the predictive distribution says
    # about points no training procedure ever saw. every number in the gap columns
    # below is a held-out score in a region with zero training support.
    x_test, y_test = sample_dataset(1200, seed=7, gap=(0.0, 0.0))
    inside_test, outside_test = region_masks(x_test)
    print(f"test points: {int(inside_test.sum())} in the gap, "
          f"{int(outside_test.sum())} on data")

    weight_decay = 1e-4
    runs = {}
    timings = {}

    # a throwaway fit before anything is timed. torch pays a one-off cost on its
    # first real backward pass, and without this the single head - which is
    # whichever method happens to go first - comes out at three times the
    # per-model cost of the ensemble members that run after it. the cost column
    # below is the argument for the cheap methods, so an artefact that inflates
    # the baseline is an artefact that flatters the conclusion.
    train_gaussian_head(x_train[:64], y_train[:64], seed=1, epochs=60, warmup=10)

    # --- the four predictive distributions ------------------------------------
    start = time.time()
    single = train_gaussian_head(x_train, y_train, seed=11)
    timings["single gaussian head"] = time.time() - start
    runs["single gaussian head"] = 1
    single_components = ensemble_components([single], x_test)

    start = time.time()
    ensemble = train_ensemble(x_train, y_train, size=ENSEMBLE_SIZE, seed_base=100)
    timings["deep ensemble (M=10)"] = time.time() - start
    runs["deep ensemble (M=10)"] = ENSEMBLE_SIZE
    ens_components = ensemble_components(ensemble, x_test)

    start = time.time()
    dropout_model = train_dropout_head(
        x_train, y_train, seed=11, rate=0.10, weight_decay=weight_decay
    )
    mc_mus, mc_sigmas = mc_dropout_samples(dropout_model, x_test, passes=DROPOUT_PASSES)
    timings["mc dropout (p=0.10)"] = time.time() - start
    runs["mc dropout (p=0.10)"] = 1
    mc_components = (mc_mus.astype(np.float64), mc_sigmas.astype(np.float64))

    start = time.time()
    laplace_model = train_dropout_head(
        x_train, y_train, seed=11, rate=0.0, weight_decay=weight_decay
    )
    covariance, _ = last_layer_laplace(
        laplace_model, x_train, y_train, prior_precision=1.0
    )
    timings["last-layer laplace"] = time.time() - start
    runs["last-layer laplace"] = 1
    lap_components = laplace_components(laplace_model, covariance, x_test)

    methods = [
        ("single gaussian head", single_components),
        ("deep ensemble (M=10)", ens_components),
        ("mc dropout (p=0.10)", mc_components),
        ("last-layer laplace", lap_components),
    ]

    scores = {name: score(mus, sigmas, x_test, y_test) for name, (mus, sigmas) in methods}

    # --- the table ------------------------------------------------------------
    print("\n--- held-out scores, lower is better ---")
    print(f"{'method':22s}{'runs':>6s}{'NLL gap':>10s}{'NLL data':>10s}"
          f"{'CRPS gap':>10s}{'CRPS data':>11s}{'cal gap':>9s}{'cal data':>10s}")
    for name, _ in methods:
        row = scores[name]
        print(f"{name:22s}{runs[name]:6d}{row['nll_gap']:10.3f}{row['nll_data']:10.3f}"
              f"{row['crps_gap']:10.4f}{row['crps_data']:11.4f}"
              f"{row['calibration_gap']:9.3f}{row['calibration_data']:10.3f}")

    print("\n--- what each method costs ---")
    for name, _ in methods:
        print(f"{name:22s}{runs[name]:3d} training run(s)   "
              f"{timings[name]:7.1f}s wall")

    # --- the two rankings -----------------------------------------------------
    by_nll = sorted(scores, key=lambda n: scores[n]["nll_gap"])
    by_crps = sorted(scores, key=lambda n: scores[n]["crps_gap"])

    print(f"\nin-gap ranking by NLL : {' < '.join(by_nll)}")
    print(f"in-gap ranking by CRPS: {' < '.join(by_crps)}")
    print(f"the two proper scores {'agree' if by_nll == by_crps else 'DISAGREE'}")

    # the day's result, asserted rather than printed and read. both of these are
    # proper scoring rules and i had been treating "proper" as though it named one
    # thing - the disagreement is not a bug in either of them, it is two different
    # questions about a predictive distribution that only coincide where there is
    # data to pin it down.
    assert by_nll[0] == "last-layer laplace", by_nll
    assert by_crps[-1] == "last-layer laplace", by_crps
    assert by_crps[0] == "deep ensemble (M=10)", by_crps

    # and the margins, so that a near-tie could not pass the two asserts above and
    # be written up as a reversal.
    nll_margin = scores[by_nll[1]]["nll_gap"] - scores[by_nll[0]]["nll_gap"]
    crps_margin = scores[by_crps[-1]]["crps_gap"] / scores[by_crps[0]]["crps_gap"]
    print(f"laplace wins NLL by {nll_margin:.2f} nats and loses CRPS by "
          f"{100 * (crps_margin - 1):.0f}%")
    assert nll_margin > 0.25, nll_margin
    assert crps_margin > 1.25, crps_margin

    # CRPS is mostly ranking the mean. stated as a claim about the two orderings
    # rather than as a correlation, because with four methods a correlation is not
    # a measurement.
    by_mean_error = sorted(scores, key=lambda n: scores[n]["mean_error_gap"])
    disagreements = sum(a != b for a, b in zip(by_crps, by_mean_error))
    print(f"CRPS ordering vs mean-error ordering: {disagreements} of 4 positions differ")
    assert disagreements <= 2, (by_crps, by_mean_error)

    # the ensemble has the best mean in the gap, which is variance reduction on the
    # mean and not uncertainty quantification. it is the reason it wins CRPS and it
    # would happen with no epistemic story attached.
    assert by_mean_error[0] == "deep ensemble (M=10)", by_mean_error

    # --- calibration curves ---------------------------------------------------
    print("\n--- empirical coverage of the central interval, in the gap ---")
    print("nominal        " + "".join(f"{lvl:7.1f}" for lvl in CALIBRATION_LEVELS))
    for name, _ in methods:
        curve = scores[name]["curve_gap"]
        print(f"{name:15s}" + "".join(f"{c:7.2f}" for c in curve))

    print("\n--- and on the data ---")
    print("nominal        " + "".join(f"{lvl:7.1f}" for lvl in CALIBRATION_LEVELS))
    for name, _ in methods:
        curve = scores[name]["curve_data"]
        print(f"{name:15s}" + "".join(f"{c:7.2f}" for c in curve))

    # calibration goes with NLL, and by a margin that makes it the least
    # ambiguous number of the four days. laplace is the only method whose central
    # intervals in the gap mean anything: nominal 0.5 covers 0.50, nominal 0.9
    # covers 0.81. the ensemble's nominal 0.9 covers 0.57 and its nominal 0.5
    # covers 0.14, and the single head's nominal 0.9 covers 0.19.
    #
    # so the only method that is honest in the gap is also the least accurate in
    # it, and it is last by the score that measures accuracy.
    best_calibrated = min(scores, key=lambda n: scores[n]["calibration_gap"])
    assert best_calibrated == "last-layer laplace", best_calibrated
    others = [
        scores[n]["calibration_gap"] for n, _ in methods if n != "last-layer laplace"
    ]
    assert scores["last-layer laplace"]["calibration_gap"] < 0.5 * min(others), (
        scores["last-layer laplace"]["calibration_gap"],
        others,
    )

    # and it is worst on the mean, which is the sentence the day turns on. asserted
    # because a version of this project where the best-calibrated method is also
    # the most accurate has no finding in it and i would like to be told.
    assert scores["last-layer laplace"]["mean_error_gap"] == max(
        scores[n]["mean_error_gap"] for n, _ in methods
    ), "the honest method stopped being the inaccurate one"

    print("\n--- the tail in the gap (99th percentile of the per-point score) ---")
    for name, _ in methods:
        row = scores[name]
        print(f"{name:22s}NLL {row['nll_worst_gap']:9.3f}   "
              f"CRPS {row['crps_worst_gap']:8.4f}   "
              f"mean |error| {row['mean_error_gap']:.4f}")

    # --- checks ---------------------------------------------------------------
    from day2_deep_ensembles import mixture_log_prob

    # the scorer against day 2's implementation, which takes models rather than
    # component arrays. two code paths for one quantity is how they drift.
    reference = -mixture_log_prob(ensemble, x_test, y_test)
    assert np.allclose(reference, mixture_nll(*ens_components, y_test), atol=1e-5)

    # CRPS of a one-component mixture is the closed-form Gaussian CRPS. the
    # mixture formula's pairwise term does not vanish at T=1 - it is E|X - X'| for
    # two draws from the same Gaussian, which is 2 sigma / sqrt(pi) - so this is a
    # real check on the constant and not a triviality.
    mu_only, sigma_only = single_components
    assert np.allclose(
        mixture_crps(mu_only, sigma_only, y_test),
        gaussian_crps(mu_only[0], sigma_only[0], y_test),
        atol=1e-9,
    )

    # CRPS against its own definition, integrated numerically on a few points.
    # the closed form is where an error would be invisible: it is smooth, it is
    # positive, it moves the right way with sigma, and it can still be wrong by a
    # constant factor.
    def crps_numeric(mu, sigma, target, points=20001, width=12.0):
        # split at the observation rather than integrating across it. the
        # integrand is `(F(x) - 1{x > y})^2`, which has a jump exactly at `y`, and
        # a trapezoid rule straddling a jump is O(h) instead of O(h^2). the first
        # version of this integrated one grid straight through and disagreed with
        # the closed form by 1e-4 - small, the right order to be discretization
        # rather than a wrong constant, and small enough that a looser tolerance
        # would have hidden it instead of explaining it.
        trapezoid = getattr(np, "trapezoid", None) or np.trapz

        left = np.linspace(target - width * sigma, target, points)
        right = np.linspace(target, target + width * sigma, points)

        below = trapezoid(_normal_cdf((left - mu) / sigma) ** 2, left)
        above = trapezoid((_normal_cdf((right - mu) / sigma) - 1.0) ** 2, right)

        return float(below + above)

    for index in (0, 17, 200, 601, 1199):
        exact = float(gaussian_crps(mu_only[0, index], sigma_only[0, index], y_test[index]))
        numeric = crps_numeric(
            float(mu_only[0, index]), float(sigma_only[0, index]), float(y_test[index])
        )
        assert abs(exact - numeric) < 1e-7, (index, exact, numeric)

    # PIT is a CDF value, and calibration on the data region is the claim day 1
    # established - the aleatoric head is right where there is data. so this curve
    # has to be near the diagonal for every method, and it is the control for the
    # gap columns: a method miscalibrated everywhere is not telling us anything
    # about the gap.
    for name, (mus, sigmas) in methods:
        pit = mixture_pit(mus, sigmas, y_test)
        assert np.all((pit >= 0.0) & (pit <= 1.0)), name
        assert scores[name]["calibration_data"] < 0.10, (
            name,
            scores[name]["calibration_data"],
        )

    # every method is worse in the gap than on the data, by both proper scores.
    # if this failed the gap would not be a gap and the whole project would be
    # measuring nothing - it is day 1's condition, restated as a held-out score
    # instead of as a curve i looked at.
    for name, _ in methods:
        row = scores[name]
        assert row["nll_gap"] > row["nll_data"], name
        assert row["crps_gap"] > row["crps_data"], name

    # on the data there is nothing to choose between them, which is the other half
    # of that design working. if the methods separated here too, the gap columns
    # would be measuring fit as much as epistemic uncertainty and none of the
    # above would be attributable.
    data_nll = [scores[n]["nll_data"] for n, _ in methods]
    data_crps = [scores[n]["crps_data"] for n, _ in methods]
    assert max(data_nll) - min(data_nll) < 0.1, data_nll
    assert max(data_crps) / min(data_crps) < 1.05, data_crps

    print("\nall checks passed")
