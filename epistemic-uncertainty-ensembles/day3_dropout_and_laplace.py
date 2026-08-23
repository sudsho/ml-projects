"""
Day 3 of epistemic uncertainty: MC dropout, and a last-layer Laplace approximation.

Day 2 left three numbers on the table. The ensemble's in-gap disagreement is 18x
the optimization floor, so it is attributable to the missing data and not to SGD
noise. It points the right way, peaking where the aleatoric term bottoms out. And
it is too small by a factor of two and a half - the worst mean error in the gap
sits at about 2.5 reported standard deviations, on a region where the model has
no information at all.

Ten training runs for that. Today's two methods both cost one.

MC dropout keeps dropout on at prediction time and treats the stochastic forward
passes as posterior samples. Gal and Ghahramani's argument is that training a
dropout network with weight decay is variational inference in a deep Gaussian
process, with the variational family being a mixture of two Gaussians per weight
row, one centred at zero. The part of that argument this file cares about is the
correspondence it implies between the dropout rate, the weight decay, the dataset
size and the model precision:

    tau = p * l**2 / (2 * N * weight_decay)

with `p` the keep probability, `l` a prior lengthscale and `N` the dataset size.
Read in the other direction, which is the direction that matters here: **the
lengthscale is implied by hyperparameters I chose before looking at the data.**
The dropout rate is a regularization knob. If it also sets the scale of the
reported epistemic uncertainty, then the number MC dropout reports in the gap is
partly a statement about my hyperparameter search and not only about the missing
data. That is testable, and testing it is most of the day.

The test is a sweep over the dropout rate with everything else fixed, each rate
run against day 2's control - the same method trained on the filled dataset, where
the interval has data in it, so whatever it reports in the gap there is that
setting's floor.

Last-layer Laplace is the other direction entirely. Take the trained network, fix
everything except the final linear layer of the mean head, and put a Gaussian
posterior on those weights: `N(w_MAP, H^-1)` with `H` the Gauss-Newton
approximation to the Hessian of the loss plus the prior precision. Because the
mean is linear in those weights, the predictive variance is exactly the Bayesian
linear regression one, `phi(x)^T H^-1 phi(x)`, in the feature space the network
learned. Closed form, no sampling, no retraining, one matrix solve.

I wrote two predictions down before running any of it and both are wrong, in
opposite directions, which is most of what this file now records.

**Prediction one: MC dropout would get the shape of the uncertainty from the data
and only its scale from the knob.** It gets both from the knob. The in-gap reading
moves with the rate, the control's in-gap reading moves with it by the same
factor, and the ratio of the two sits at 1.0 across the whole sweep. At `p = 0.20`
and `p = 0.35` it is *below* one - the ensemble trained on data with no hole in it
reports more in-gap disagreement than the one trained on data with a hole. MC
dropout is not detecting the missing data at all. It is reporting the variance the
masks inject, which is present everywhere in equal measure.

The number that would have fooled me is `0.1465` at `p = 0.10`. Next to the
ensemble's `0.2906` that is half as large and the same order, exactly what a
slightly-worse-but-working method looks like, and it means nothing, because the
filled-data control returns `0.1381` for the same setting. Day 2 built that
control and the control passed. This is the first time it has killed something,
and what it killed was not an outlier or a bug - it was an ordinary-looking number
I would have written up.

**Prediction two: last-layer Laplace would under-report in the gap, because the
feature map is frozen and the gap might not be a hole in feature space.** It is
the best of the three on every column - a ratio of 19.4 against the ensemble's
10.3, a signal of 48x against its own control, and a miss of 1.11, the only method
whose reported spread comes close to covering its own error.

The frozen-feature objection is a real one and I imported it without checking that
its premise held here. It is an argument about high-dimensional inputs, where the
data manifold is thin and a point far from it in input space can still land on
covered features. This input is one-dimensional: `phi(x)` traces a curve through
`R^64`, the training data covers two arcs of that curve, and the gap is the arc
between them. Because the target is curvy enough that day 1 had to widen the gap
to stop the network interpolating through it, that middle arc is somewhere the
training features genuinely do not go, and the quadratic form has plenty to be
large about.

What survives of the hyperparameter argument is that it moves house. It does not
land on MC dropout, whose reading is a knob all the way down. It lands on Laplace,
via the prior precision: two decades of it move the in-gap standard deviation by
6x and the ratio by 5x. The method that works also has a free parameter setting
how much it works.
"""

import numpy as np
import torch
import torch.nn as nn

from day1_the_gap import (
    DOMAIN,
    GAP,
    decompose,
    gaussian_nll,
    predict,
    sample_dataset,
    true_mean,
    true_sigma,
)
from day2_deep_ensembles import region_masks


DROPOUT_PASSES = 200


class DropoutGaussianHead(nn.Module):
    """`GaussianHead` with dropout between the body's layers.

    Same width, same activation, same clamped `log_var` head. The only structural
    change is `nn.Dropout` after each tanh, and the only behavioural change is
    that `forward` can be asked to keep dropping at prediction time.

    Dropout goes in the body and not on the heads on purpose. The body is the
    feature map, which is the thing whose uncertainty the whole project is about;
    dropping units in the `log_var` head would add spread to the *aleatoric*
    estimate, which would land in the wrong term of the decomposition and would
    look like the method working.
    """

    def __init__(self, hidden=64, rate=0.1, log_var_range=(-8.0, 2.0)):
        super().__init__()
        self.rate = rate
        self.first = nn.Linear(1, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(rate)
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        self.log_var_range = log_var_range

    def features(self, x, stochastic):
        h = self.drop_maybe(torch.tanh(self.first(x)), stochastic)
        return self.drop_maybe(torch.tanh(self.second(h)), stochastic)

    def drop_maybe(self, h, stochastic):
        """Dropout that ignores `self.training` when asked to.

        `nn.Dropout` is a no-op under `model.eval()`, which is what makes MC
        dropout an easy thing to get silently wrong: calling `predict` on an
        eval-mode model returns the same value every pass, the variance over
        passes is exactly zero, and nothing raises. Routing every call through
        one place with an explicit flag means the choice is visible at the call
        site instead of being a property of the model's mode.
        """
        if not stochastic or self.rate == 0.0:
            return h
        return torch.nn.functional.dropout(h, p=self.rate, training=True)

    def forward(self, x, stochastic=False):
        h = self.features(x, stochastic)
        mu = self.mean_head(h).squeeze(-1)
        log_var = self.log_var_head(h).squeeze(-1).clamp(*self.log_var_range)
        return mu, log_var


def train_dropout_head(
    x, y, seed, rate=0.1, epochs=3000, warmup=300, lr=1e-2, hidden=64, weight_decay=1e-4
):
    """Fit a `DropoutGaussianHead`, same recipe as day 1 plus weight decay.

    The MSE warm-up is kept for day 1's reason - joint NLL training from scratch
    lets the variance head grow to cover the residuals of an untrained mean and
    then the gradient on the mean flattens.

    Weight decay is not decoration here. It is the `lambda` in the lengthscale
    correspondence, so a run with `weight_decay=0` does not have an implied
    lengthscale at all and the dropout-as-VI reading does not apply to it. Using
    Adam rather than AdamW means the decay enters as an L2 term in the gradient
    rather than as a separate shrink, which is the form the correspondence is
    written for.
    """
    torch.manual_seed(seed)

    model = DropoutGaussianHead(hidden=hidden, rate=rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    targets = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        mu, log_var = model(inputs, stochastic=True)

        if epoch < warmup:
            loss = ((mu - targets) ** 2).mean()
        else:
            loss = gaussian_nll(mu, log_var, targets).mean()

        loss.backward()
        optimizer.step()

    return model


def mc_dropout_samples(model, x, passes=DROPOUT_PASSES, seed=0):
    """`passes` stochastic forward passes - returns `(means, sigmas)`, both `(T, n)`.

    Each pass uses a different dropout mask and is one sample from the implied
    approximate posterior. The seed is fixed so the reported numbers are
    reproducible, which matters more here than usual: the whole day is about
    whether a quantity is a property of the data or of a setting, and a quantity
    that moves between runs is neither.
    """
    torch.manual_seed(seed)
    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)

    means, sigmas = [], []
    with torch.no_grad():
        for _ in range(passes):
            mu, log_var = model(inputs, stochastic=True)
            means.append(mu.numpy())
            sigmas.append((0.5 * log_var).exp().numpy())

    return np.stack(means), np.stack(sigmas)


def decompose_samples(means, sigmas):
    """Law of total variance over stacked samples rather than over a list of models.

    `decompose` in day 1 takes models and calls `predict` on each; MC dropout's
    samples come from one model and there is nothing to put in that list. Same
    arithmetic, different input type, and the `__main__` block asserts the two
    agree on an ensemble so this is not a second implementation drifting from the
    first.

    `ddof=0` for day 2's reason. The law of total variance for a uniform mixture
    is exact with the population variance over the components, and the mixture is
    what gets scored. The correction belongs where the members are being treated
    as a sample from a posterior, which is what `epistemic_by_size` asked about.
    """
    return means.mean(axis=0), (sigmas ** 2).mean(axis=0), means.var(axis=0)


def implied_lengthscale(rate, tau, n_train, weight_decay):
    """Invert `tau = p * l**2 / (2 * N * lambda)` for `l`, with `p` the keep rate.

    The correspondence from Gal and Ghahramani, read backwards. Given the model
    precision the fitted network actually reports, this is the prior lengthscale
    the dropout rate and weight decay are implicitly asserting.

    It is here to be *read*, not used - nothing downstream consumes it. The point
    is that a number with units of input distance falls out of two settings that
    were chosen for optimization reasons, and the epistemic uncertainty in the gap
    is being compared against a gap width of 3.0 in those same units.
    """
    keep = 1.0 - rate
    return float(np.sqrt(2.0 * n_train * weight_decay * tau / keep))


def last_layer_laplace(model, x, y, prior_precision=1.0):
    """Gaussian posterior over the mean head's weights, via the Gauss-Newton Hessian.

    The mean is `mu(x) = w . phi(x) + b` with `phi` the body's output, so with the
    body frozen the model is a Bayesian linear regression in `phi`. Under the
    Gaussian likelihood with a per-point variance `sigma_i**2` from the other head,
    the Gauss-Newton Hessian of the negative log posterior is

        H = sum_i phi_i phi_i^T / sigma_i**2 + prior_precision * I

    which is exact for this loss rather than an approximation, because `mu` is
    linear in `w` and the loss is quadratic in `mu`. The Gauss-Newton name is kept
    because the general recipe is the approximation and this is the case where it
    stops being one - the dropped term is the second derivative of `mu` with
    respect to `w`, and that is zero here.

    `sigma_i` is treated as fixed and known rather than as a second thing to be
    uncertain about. That is a real restriction and it is the right one for this
    project: the aleatoric term is what day 1 established the network gets right,
    and putting a posterior on it as well would mix the two terms back together
    in the one place the whole design is trying to keep them apart.

    Features are augmented with a constant so the bias is in the posterior too.
    Leaving it out puts a hard zero-variance constraint on the intercept, which
    shows up as an epistemic term that is too small by a constant everywhere -
    small, and in the direction that would flatter the method.

    Returns `(covariance, prior_precision)` with covariance `(hidden+1, hidden+1)`.
    """
    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        if isinstance(model, DropoutGaussianHead):
            features = model.features(inputs, stochastic=False)
            log_var = model.log_var_head(features).squeeze(-1)
            log_var = log_var.clamp(*model.log_var_range)
        else:
            features = model.body(inputs)
            log_var = model.log_var_head(features).squeeze(-1)
            log_var = log_var.clamp(*model.log_var_range)

    phi = features.numpy().astype(np.float64)
    phi = np.concatenate([phi, np.ones((len(phi), 1))], axis=1)
    variance = np.exp(log_var.numpy().astype(np.float64))

    hessian = (phi / variance[:, None]).T @ phi
    hessian += prior_precision * np.eye(phi.shape[1])

    # solve rather than invert, and symmetrize afterwards. the Hessian is
    # positive definite by construction, but `phi` has 64 tanh columns fitted on
    # 600 points and several of them are close to collinear, so the condition
    # number is large and the asymmetry that comes back from a naive inverse is
    # big enough to make the quadratic form below go slightly negative.
    covariance = np.linalg.solve(hessian, np.eye(phi.shape[1]))
    covariance = 0.5 * (covariance + covariance.T)

    return covariance, prior_precision


def laplace_predict(model, covariance, x):
    """`(mu, aleatoric_var, epistemic_var)` under the last-layer posterior.

    The epistemic term is `phi(x)^T Sigma phi(x)`, the pushforward of the weight
    posterior through a map that is linear in the weights, so it is exact given
    the posterior rather than a delta-method approximation.

    The aleatoric term comes from the `log_var` head unchanged, which keeps the
    decomposition comparable with the ensemble's - both report the same thing in
    that slot and differ only in the other one.
    """
    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        if isinstance(model, DropoutGaussianHead):
            features = model.features(inputs, stochastic=False)
        else:
            features = model.body(inputs)
        mu = model.mean_head(features).squeeze(-1).numpy()
        log_var = model.log_var_head(features).squeeze(-1)
        log_var = log_var.clamp(*model.log_var_range).numpy()

    phi = features.numpy().astype(np.float64)
    phi = np.concatenate([phi, np.ones((len(phi), 1))], axis=1)

    epistemic = np.einsum("ij,jk,ik->i", phi, covariance, phi)

    return mu, np.exp(log_var), epistemic


def region_report(mean, aleatoric, epistemic, x_grid):
    """Regional averages in standard-deviation units, plus the mean's error.

    Same fields as day 2's `summarize` so the three methods land in one table.
    Standard deviations rather than variances throughout, because everything is
    eventually compared against a noise scale and against a mean error, and both
    of those are unsquared.
    """
    inside, outside = region_masks(x_grid)
    epistemic = np.maximum(epistemic, 0.0)

    return {
        "epistemic_sd_gap": float(np.sqrt(epistemic[inside]).mean()),
        "epistemic_sd_data": float(np.sqrt(epistemic[outside]).mean()),
        "aleatoric_sd_gap": float(np.sqrt(aleatoric[inside]).mean()),
        "aleatoric_sd_data": float(np.sqrt(aleatoric[outside]).mean()),
        "mean_error_gap": float(np.abs(mean - true_mean(x_grid))[inside].max()),
        "ratio": float(
            np.sqrt(epistemic[inside]).mean() / np.sqrt(epistemic[outside]).mean()
        ),
        "miss": float(
            np.abs(mean - true_mean(x_grid))[inside].max()
            / np.sqrt(epistemic[inside]).mean()
        ),
    }


if __name__ == "__main__":
    torch.set_num_threads(1)

    grid = np.linspace(DOMAIN[0], DOMAIN[1], 400).astype(np.float32)
    inside, outside = region_masks(grid)

    x_train, y_train = sample_dataset(600, seed=0)
    filled_n = int(round(600 * 8.0 / 5.0))
    x_filled, y_filled = sample_dataset(filled_n, seed=0, gap=(0.0, 0.0))

    weight_decay = 1e-4

    # --- MC dropout, and the sweep that is the point of it -------------------
    print("--- mc dropout, by rate ---")
    print(f"{'rate':>6s}{'gap sd':>10s}{'data sd':>10s}{'ratio':>8s}"
          f"{'miss':>8s}{'floor':>10s}{'signal':>9s}{'lengthscale':>13s}")

    sweep = {}
    for rate in (0.05, 0.10, 0.20, 0.35):
        model = train_dropout_head(
            x_train, y_train, seed=11, rate=rate, weight_decay=weight_decay
        )
        means, sigmas = mc_dropout_samples(model, grid)
        report = region_report(*decompose_samples(means, sigmas), grid)

        # the control, exactly as day 2 defines it: the same method on data with
        # the interval filled in. whatever it reports in the gap there is this
        # method's floor, because the missing data is no longer missing.
        control = train_dropout_head(
            x_filled, y_filled, seed=11, rate=rate, weight_decay=weight_decay
        )
        control_report = region_report(
            *decompose_samples(*mc_dropout_samples(control, grid)), grid
        )
        floor = control_report["epistemic_sd_gap"]

        tau = 1.0 / float(np.mean(sigmas ** 2))
        lengthscale = implied_lengthscale(rate, tau, len(x_train), weight_decay)

        sweep[rate] = (report, floor, lengthscale)
        print(f"{rate:6.2f}{report['epistemic_sd_gap']:10.4f}"
              f"{report['epistemic_sd_data']:10.4f}{report['ratio']:8.2f}"
              f"{report['miss']:8.2f}{floor:10.4f}"
              f"{report['epistemic_sd_gap'] / floor:9.1f}{lengthscale:13.4f}")

    # the day's first result, and it is not the one the sweep was built to get.
    #
    # i expected the magnitude to move with the rate and the in-gap / on-data
    # ratio to stay put - shape from the data, scale from a knob. the magnitude
    # does move, 1.8x across the sweep. so does the control's floor, by the same
    # kind of factor, and the ratio of the two never leaves 1.
    #
    # so there is no shape to separate from the scale. against its own control mc
    # dropout is reporting nothing about the gap at any rate, and at p=0.20 and
    # p=0.35 it reports *less* in-gap disagreement than the model trained on data
    # with no gap in it. what it measures is the variance the masks inject, which
    # is a property of the mask distribution and is there everywhere in equal
    # measure whether or not there is data underneath.
    magnitudes = [sweep[r][0]["epistemic_sd_gap"] for r in sorted(sweep)]
    floors = [sweep[r][1] for r in sorted(sweep)]
    signals = [m / f for m, f in zip(magnitudes, floors)]

    magnitude_spread = max(magnitudes) / min(magnitudes)
    floor_spread = max(floors) / min(floors)

    print(f"\nacross the rate sweep: magnitude moves {magnitude_spread:.1f}x, "
          f"control floor moves {floor_spread:.1f}x, "
          f"signal stays in [{min(signals):.1f}, {max(signals):.1f}]")

    assert magnitude_spread > 1.5, (
        "the dropout rate does not move the reported magnitude at all - the "
        "sweep has nothing in it"
    )

    # the floor tracks the magnitude, which is the whole finding. if the rate
    # were setting a scale on top of a real signal these two would come apart.
    assert 0.5 < floor_spread / magnitude_spread < 2.0, (
        magnitude_spread,
        floor_spread,
    )

    # and the signal is absent at every rate. day 2's ensemble cleared its floor
    # by 18x; this clears nothing. asserted as an upper bound, which is the
    # uncomfortable direction to write an assert in and is the correct one here -
    # the claim is that the method fails the control, so the test has to fail if
    # it ever passes.
    for rate in sorted(sweep):
        report, floor, _ = sweep[rate]
        assert report["epistemic_sd_gap"] < 2.0 * floor, (
            rate,
            report["epistemic_sd_gap"],
            floor,
        )

    # the implied lengthscale is about 1.1 at every rate against a gap of width
    # 3.0. so the prior the correspondence attributes to this model already says
    # the two sides of the hole are nearly three correlation lengths apart, and
    # the method reports nothing between them anyway. the number is here to be
    # read and nothing consumes it; what it says is that the failure above is not
    # the correspondence being violated - it is the posterior samples not being
    # samples from the thing the correspondence describes.
    lengthscales = [sweep[r][2] for r in sorted(sweep)]
    print(f"implied prior lengthscale {min(lengthscales):.2f} to "
          f"{max(lengthscales):.2f}, against a gap of width {GAP[1] - GAP[0]:.1f}")

    # --- last-layer Laplace ---------------------------------------------------
    # one training run, no sampling, one solve of a 65x65 system.
    print("\n--- last-layer laplace ---")

    laplace_model = train_dropout_head(
        x_train, y_train, seed=11, rate=0.0, weight_decay=weight_decay
    )
    covariance, prior_precision = last_layer_laplace(
        laplace_model, x_train, y_train, prior_precision=1.0
    )
    laplace = region_report(*laplace_predict(laplace_model, covariance, grid), grid)

    control_model = train_dropout_head(
        x_filled, y_filled, seed=11, rate=0.0, weight_decay=weight_decay
    )
    control_cov, _ = last_layer_laplace(
        control_model, x_filled, y_filled, prior_precision=1.0
    )
    laplace_floor = region_report(
        *laplace_predict(control_model, control_cov, grid), grid
    )["epistemic_sd_gap"]

    print(f"epistemic sd  in gap {laplace['epistemic_sd_gap']:.4f}   "
          f"on data {laplace['epistemic_sd_data']:.4f}   "
          f"ratio {laplace['ratio']:.2f}")
    print(f"control floor in gap {laplace_floor:.4f}   "
          f"signal {laplace['epistemic_sd_gap'] / laplace_floor:.1f}x")
    print(f"miss (in-gap error / reported sd) {laplace['miss']:.2f}")

    # the posterior is a posterior. positive definite covariance, and a quadratic
    # form that never comes out negative on the grid. the symmetrization in
    # last_layer_laplace is what makes the second of those safe rather than
    # nearly-safe: phi has 64 tanh columns fitted on 600 points, several close to
    # collinear, so the solve comes back with an asymmetry that a quadratic form
    # can turn into a small negative number at the grid points where the form is
    # smallest.
    assert np.all(np.linalg.eigvalsh(covariance) > 0), "covariance is not PD"
    _, _, laplace_epistemic = laplace_predict(laplace_model, covariance, grid)
    assert np.all(laplace_epistemic >= 0.0), "negative predictive variance"

    # --- the three methods on one grid ---------------------------------------
    from day2_deep_ensembles import train_ensemble

    print("\ntraining the day-2 ensemble for the comparison ...")
    ensemble = train_ensemble(x_train, y_train, size=10, seed_base=100)
    ens_mean, ens_aleatoric, ens_epistemic = decompose(ensemble, grid)
    ens = region_report(ens_mean, ens_aleatoric, ens_epistemic, grid)

    mc_report, mc_floor, _ = sweep[0.10]

    print("\n--- epistemic sd, one row per method ---")
    print(f"{'method':22s}{'runs':>6s}{'in gap':>10s}{'on data':>10s}"
          f"{'ratio':>8s}{'miss':>8s}")
    rows = (
        ("deep ensemble (M=10)", 10, ens),
        ("mc dropout (p=0.10)", 1, mc_report),
        ("last-layer laplace", 1, laplace),
    )
    for label, runs, row in rows:
        print(f"{label:22s}{runs:6d}{row['epistemic_sd_gap']:10.4f}"
              f"{row['epistemic_sd_data']:10.4f}{row['ratio']:8.2f}"
              f"{row['miss']:8.2f}")

    # the day's second result, and the other one i got backwards.
    #
    # i predicted laplace would under-report in the gap because the feature map is
    # frozen: the gap is a hole in input space, and whether it is a hole in
    # *feature* space is a separate question that a posterior over the last layer
    # alone cannot answer. it separates the regions nearly twice as sharply as the
    # ensemble does.
    #
    # the frozen-feature objection is real and i imported it without checking its
    # premise. it is an argument about high-dimensional inputs, where the data
    # manifold is thin and a point far away in input space can still land on
    # covered features. this input is one-dimensional. phi(x) traces a curve
    # through R^64, the training data covers two arcs of it, and the gap is the
    # arc between them - which is genuinely uncovered, because day 1 widened the
    # gap until the target stopped being interpolable and that is the same
    # condition as the feature curve going somewhere new.
    #
    # so the objection is not wrong, it is about a regime this experiment is not
    # in, and one dimension is the regime where the last layer is enough.
    print(f"\nratio: ensemble {ens['ratio']:.2f}, laplace {laplace['ratio']:.2f}, "
          f"mc dropout {mc_report['ratio']:.2f}")
    assert laplace["ratio"] > ens["ratio"], (
        "last-layer laplace separated the regions worse than the ensemble, which "
        "is what the frozen-feature-map argument predicted and is not what ran"
    )
    assert mc_report["ratio"] < 2.0, mc_report["ratio"]

    # all three still under-report, which is day 2's finding surviving contact
    # with two more methods. none of the misses is below 1, so no method's
    # reported spread covers its own error in the gap - though laplace at 1.11 is
    # close enough that the gap between "knows it does not know" and "knows how
    # much it does not know" has nearly closed for one of them.
    for label, _, row in rows:
        assert row["miss"] > 1.0, (label, row["miss"])

    assert laplace["miss"] < 1.5, laplace["miss"]
    assert mc_report["miss"] > 4.0, mc_report["miss"]

    # --- the shared arithmetic is shared -------------------------------------
    # decompose_samples against day 1's decompose, on the same ensemble. two
    # implementations of the law of total variance, one taking models and one
    # taking stacked arrays, and this is the only thing keeping them equal.
    stacked_means = np.stack([predict(m, grid)[0] for m in ensemble])
    stacked_sigmas = np.stack([predict(m, grid)[1] for m in ensemble])
    check_mean, check_aleatoric, check_epistemic = decompose_samples(
        stacked_means, stacked_sigmas
    )
    assert np.allclose(check_mean, ens_mean)
    assert np.allclose(check_aleatoric, ens_aleatoric)
    assert np.allclose(check_epistemic, ens_epistemic)

    # the aleatoric term is the same story in all three, which is what makes the
    # epistemic column comparable at all. it has to point the other way from the
    # epistemic one - true_sigma is smallest at the centre - and a method
    # re-reporting the aleatoric term under a different name gets that backwards.
    for label, _, row in rows:
        assert row["aleatoric_sd_gap"] < row["aleatoric_sd_data"], label

    # --- how many passes before mc dropout stops moving ----------------------
    # day 2 asked this of the ensemble size and found the estimate still climbing
    # at ten members. the analogous question here is cheap, because passes cost a
    # forward evaluation instead of a training run, and the answer should be the
    # opposite - this is a Monte Carlo error over one fixed posterior, not a
    # sample size over a distribution i cannot draw from.
    print("\n--- mc dropout, epistemic sd in gap by pass count ---")
    model = train_dropout_head(x_train, y_train, seed=11, rate=0.10,
                               weight_decay=weight_decay)
    by_passes = []
    for passes in (10, 25, 50, 100, 200, 400):
        values = []
        for repeat in range(5):
            means, sigmas = mc_dropout_samples(model, grid, passes=passes,
                                               seed=1000 + repeat)
            _, _, epistemic = decompose_samples(means, sigmas)
            values.append(float(np.sqrt(np.maximum(epistemic, 0.0))[inside].mean()))
        by_passes.append((passes, float(np.mean(values)), float(np.std(values))))
        print(f"{passes:6d}{by_passes[-1][1]:12.4f}{by_passes[-1][2]:12.4f}"
              f"{by_passes[-1][2] / by_passes[-1][1]:10.1%}")

    # and it converges, tightly. the relative spread at 400 passes is well under
    # a percent, against the ensemble's near-30% at M=8 from day 2 - which is the
    # difference between Monte Carlo error over a fixed posterior, where more
    # samples are a forward pass each, and sample size over a posterior i cannot
    # draw from, where more samples are a training run each.
    #
    # it is also the best-behaved estimator in the day, estimating the quantity
    # the day just established is the wrong one. converging quickly to a number
    # that fails its own control is worth stating in that order, because the
    # convergence is the part that looks like evidence.
    assert by_passes[-1][2] / by_passes[-1][1] < 0.02, (
        "mc dropout's estimate is still noisy at 400 passes"
    )
    assert by_passes[-1][2] < by_passes[0][2], "more passes did not help"

    # --- the knob, on the method that works ----------------------------------
    # the hyperparameter argument this day was built around, landing somewhere
    # other than where it was aimed. mc dropout's reading is a knob all the way
    # down and so there is nothing to separate; laplace's is a real signal with a
    # free scale on it, and two decades of prior precision move the in-gap
    # standard deviation by about 6x and the ratio by about 5x.
    print(f"\nprior precision sensitivity, laplace:")
    precision_rows = []
    for precision in (0.1, 1.0, 10.0):
        cov, _ = last_layer_laplace(laplace_model, x_train, y_train, precision)
        row = region_report(*laplace_predict(laplace_model, cov, grid), grid)
        precision_rows.append(row)
        print(f"  prior_precision={precision:5.1f}  "
              f"gap sd {row['epistemic_sd_gap']:.4f}  ratio {row['ratio']:.2f}  "
              f"miss {row['miss']:.2f}")

    gap_sds = [row["epistemic_sd_gap"] for row in precision_rows]
    assert max(gap_sds) / min(gap_sds) > 4.0, gap_sds

    # so the epistemic standard deviation is not a scoreable quantity across
    # methods, for three different reasons: mc dropout's is set by the mask rate,
    # laplace's by the prior precision, and the ensemble's by how concentrated
    # SGD-from-a-gaussian-init happens to be relative to the posterior - which is
    # not a knob, but is not a principled scale either.
    #
    # day 4 therefore scores the predictive distributions rather than the
    # decomposition. NLL and CRPS both punish over-confidence and
    # under-confidence, so no method can buy a score by inflating its
    # uncertainty, and inflating its uncertainty is exactly the degree of freedom
    # this day found in two of the three.
    #
    # and day 4 keeps the control on every number it reports. it cost one extra
    # training run per setting today and it is the only reason the mc dropout
    # column is not in the writeup as a working method.
    print("\nday 3 done")
