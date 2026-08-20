"""
Day 1 of epistemic uncertainty.

The MDN project finished with a model that reports the shape of the conditional
distribution - three modes where there are three, mass on every branch, PIT
calibration that passes in the pooled test and fails in each half of the input
range separately. Everything it says is about the *spread of the data*. Nothing
it says is about whether it has seen data here at all.

That is not a defect in the fit. It is what negative log-likelihood asks for. The
mixture is trained to match `p(t | y)` wherever the training `y` landed, and at a
`y` that never appeared in training the network still emits a perfectly ordinary
set of mixture parameters, produced by whatever the hidden layers extrapolate to,
with no marker on them saying they were made up. An MDN is as confident outside
its data as inside it, and the confidence means the same thing in both places,
which is to say it means the first thing and not the second.

The two quantities have names. Split the predictive variance with the law of
total variance over the posterior on weights:

    Var[y | x]  =  E_theta[ sigma^2(x, theta) ]  +  Var_theta[ mu(x, theta) ]
                   \-------- aleatoric --------/    \------- epistemic ------/

The first term is noise in the data-generating process: irreducible, real, and
the only one an MDN or a heteroscedastic Gaussian head models. The second is
disagreement between the models the data has not ruled out. It is reducible - it
shrinks when data arrives - and it is the one that is supposed to grow where
there is no data.

**A single network sets the second term to zero identically.** Not approximately,
not badly - structurally. One point estimate of `theta` means `Var_theta` is a
variance over a point mass. So the honest statement of today is not that a single
heteroscedastic network is poorly calibrated off-distribution; it is that it has
no slot for the quantity, in exactly the sense that the MDN project's day 1 point
regressor had one output slot and needed three. Same failure, one level up: the
wrong output object rather than the wrong number.

Today builds the instrument that will measure everything in the next three days:

  - a regression dataset with a deliberate hole punched in its input range, and
    known heteroscedastic noise, so both terms of the decomposition have a ground
    truth to be scored against;
  - a Gaussian head trained by NLL, which recovers the aleatoric term where there
    is data - the point being that this half works, so any failure in the gap is
    attributable to the missing half rather than to a bad fit;
  - the measurement in the gap: the reported sigma does not grow, the mean is
    wrong, and the standardized residuals blow up;
  - and a preview of the whole project, two identical networks from different
    initializations, agreeing where there is data and disagreeing where there is
    none. That disagreement is the epistemic term, empirically, and whether it is
    a usable signal or mostly initialization noise is day 2's question and not a
    thing to assume today.

One thing to be careful about from the start, because it decides whether any of
this means anything: "the reported sigma in the gap" is not a number the model
computed about the gap. It is whatever the network's smooth interpolation between
the two data regions happens to produce there. It is a function of the
architecture and the activation, not of the data, and if it came out large that
would be an accident and not a signal. So the test below is not "sigma is small
in the gap" - it is that sigma in the gap sits inside the range of values sigma
takes on the training data, which is the statement that nothing distinguishes
the two.
"""

import numpy as np
import torch
import torch.nn as nn


GAP = (-1.5, 1.5)
DOMAIN = (-4.0, 4.0)


def true_mean(x):
    """The regression function `f(x) = sin(2x) + 0.3x`.

    Curved enough that the two data regions do not pin down the middle by
    accident - a straight line through both halves would make the gap trivially
    interpolable and there would be nothing for epistemic uncertainty to be about.

    The gap width was set by this and not chosen first. At `(-1, 1)` the hole
    holds barely more than one turning point of `sin(2x)`, a tanh network
    interpolates straight through it, and the worst error in the middle came out
    at 0.42 - under two noise scales, so the mean was not visibly wrong and there
    was nothing for the reported sigma to fail to cover. Widening to `(-1.5, 1.5)`
    puts most of a full period inside the hole, with a minimum near `-0.79` and a
    maximum near `0.79` that nothing in the data implies. That is the honest
    version of "the answer in the gap is unconstrained", and it is worth recording
    that the first attempt was not: a gap you can interpolate is not a gap, and if
    it had gone unnoticed every measurement in the next three days would have been
    made on a problem with no epistemic uncertainty in it.
    """
    return np.sin(2.0 * x) + 0.3 * x


def true_sigma(x):
    """Aleatoric noise scale, `0.05 + 0.10 |x|`.

    Heteroscedastic on purpose and increasing outward, which is the awkward
    direction: it means the noise is *largest* far from the gap and smallest at
    its edges. A model that confuses the two kinds of uncertainty gets the sign
    of the effect wrong here, since the epistemic term is supposed to peak in the
    middle where the aleatoric term is at its minimum. Making them anti-correlated
    is the only way to tell later whether a method is finding the second one or
    re-reporting the first.
    """
    return 0.05 + 0.10 * np.abs(x)


def sample_dataset(n, seed, gap=GAP, domain=DOMAIN):
    """Draw `n` points from `domain` with the interval `gap` removed.

    Rejection rather than two uniforms glued together, so the density is exactly
    uniform on what remains and the two sides get their share by area instead of
    by a hand-set ratio.
    """
    rng = np.random.default_rng(seed)

    xs = []
    while len(xs) < n:
        draw = rng.uniform(domain[0], domain[1], size=n)
        xs.extend(draw[(draw <= gap[0]) | (draw >= gap[1])].tolist())

    x = np.array(xs[:n])
    y = true_mean(x) + true_sigma(x) * rng.standard_normal(n)

    return x.astype(np.float32), y.astype(np.float32)


class GaussianHead(nn.Module):
    """MLP emitting `(mu, log_var)` - a conditional Gaussian, not a point.

    The MDN emitted `3K` numbers for a `K`-component mixture; this is that with
    `K = 1`, which is the right baseline for this project because the question
    has moved. Multimodality is settled and not what is being studied here. What
    matters is that the head can express an input-dependent *width*, so that the
    aleatoric term is modelled properly and any remaining failure has to be the
    other term.

    `log_var` rather than `sigma` for the usual reason - it is unconstrained, so
    the optimizer never has to be kept on the positive side of a boundary - and
    it is clamped, which is not decoration. The NLL rewards shrinking the variance
    on any point the mean happens to fit well, and early in training that is a
    handful of points at initialization; without a floor the loss runs off to
    minus infinity on those and the gradient signal from everything else is lost.
    """

    def __init__(self, hidden=64, log_var_range=(-8.0, 2.0)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        self.log_var_range = log_var_range

    def forward(self, x):
        features = self.body(x)
        mu = self.mean_head(features).squeeze(-1)
        log_var = self.log_var_head(features).squeeze(-1)
        log_var = log_var.clamp(*self.log_var_range)
        return mu, log_var


def gaussian_nll(mu, log_var, target):
    """Negative log-likelihood of `target` under `N(mu, exp(log_var))`, per point.

    Constant dropped. Written out rather than taken from `nn.GaussianNLLLoss`
    because the two terms are the whole point of the day and they should be
    visible: `log_var` is the price of claiming to be uncertain, and the
    normalized square is the price of being wrong. The model trades them off, and
    that trade is exactly what it cannot do in the gap - there is no residual
    there to pay for a wider sigma, so nothing pushes it wider.
    """
    return 0.5 * (log_var + (target - mu) ** 2 / log_var.exp())


def train_gaussian_head(x, y, seed, epochs=3000, warmup=300, lr=1e-2, hidden=64):
    """Fit a `GaussianHead` by NLL, with an MSE warm-up on the mean.

    The warm-up is not a nicety. Joint NLL training from scratch has a bad
    attractor: the variance head can grow to cover the residuals of an untrained
    mean, which lowers the loss immediately and then flattens the gradient on the
    mean head, since the normalized square is divided by a now-large variance. The
    model settles for explaining the data as noise. Fitting the mean under plain
    squared error first puts the residuals near their irreducible size before the
    variance head is allowed to have an opinion about them.

    Full batch. The dataset is 400 one-dimensional points and minibatching would
    add gradient noise to a study whose entire subject is the difference between
    two sources of variance.
    """
    torch.manual_seed(seed)

    model = GaussianHead(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    targets = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        mu, log_var = model(inputs)

        if epoch < warmup:
            loss = ((mu - targets) ** 2).mean()
        else:
            loss = gaussian_nll(mu, log_var, targets).mean()

        loss.backward()
        optimizer.step()

    return model


def predict(model, x):
    """Return `(mu, sigma)` as numpy arrays for a 1-D array of inputs."""
    model.eval()
    with torch.no_grad():
        mu, log_var = model(torch.tensor(x, dtype=torch.float32).unsqueeze(-1))

    return mu.numpy(), (0.5 * log_var).exp().numpy()


def decompose(models, x):
    """Split predictive variance into aleatoric and epistemic parts.

    The law of total variance over whatever ensemble `models` represents:

        E[sigma^2]  (mean of the per-model variances)     -> aleatoric
        Var[mu]     (variance of the per-model means)     -> epistemic

    With one model in the list the second term is exactly zero, which is the point
    of running it that way today rather than a degenerate case to be avoided. It
    is the same function that day 2 calls with an ensemble in it, so the zero is
    produced by the definition rather than asserted in the text.
    """
    means = np.stack([predict(m, x)[0] for m in models])
    sigmas = np.stack([predict(m, x)[1] for m in models])

    aleatoric = (sigmas ** 2).mean(axis=0)
    epistemic = means.var(axis=0)

    return means.mean(axis=0), aleatoric, epistemic


if __name__ == "__main__":
    x_train, y_train = sample_dataset(600, seed=0)

    assert not ((x_train > GAP[0]) & (x_train < GAP[1])).any()
    print(f"train: n={len(x_train)}  "
          f"left={(x_train <= GAP[0]).sum()}  right={(x_train >= GAP[1]).sum()}")

    model = train_gaussian_head(x_train, y_train, seed=1)

    # --- the half that works -------------------------------------------------
    #
    # scored on held-out points from the same distribution, so this says the fit
    # is good rather than that the fit memorized. if the aleatoric half did not
    # come out here there would be nothing to attribute the gap behaviour to.
    x_test, y_test = sample_dataset(600, seed=7)
    mu_test, sigma_test = predict(model, x_test)

    sigma_error = np.abs(sigma_test - true_sigma(x_test)) / true_sigma(x_test)
    print(f"\nin-distribution, held out (n={len(x_test)})")
    print(f"  mean |mu - f(x)|          : {np.abs(mu_test - true_mean(x_test)).mean():.4f}")
    print(f"  median relative sigma err : {np.median(sigma_error):.4f}")

    # the mean is accurate to well under one noise scale, and the reported sigma
    # tracks the true one rather than collapsing to a constant. a homoscedastic
    # model would pass the first check and fail the second.
    assert np.abs(mu_test - true_mean(x_test)).mean() < 0.5 * true_sigma(x_test).mean()
    assert np.median(sigma_error) < 0.35, np.median(sigma_error)

    # heteroscedasticity actually learned, not faked by a constant that happens to
    # sit near the average. two checks rather than one: the correlation says the
    # trend is there, the outer-versus-inner ratio says it has the right *size*,
    # and only the second one would notice a model that got the direction right
    # and the magnitude half. worth keeping both, because a plausible failure in
    # the next three days is a method that reports something shaped like
    # uncertainty and scaled like nothing in particular.
    corr = np.corrcoef(sigma_test, np.abs(x_test))[0, 1]
    outer = sigma_test[np.abs(x_test) > 3.0].mean()
    inner = sigma_test[np.abs(x_test) < 2.5].mean()
    print(f"  corr(sigma_hat, |x|)      : {corr:.4f}")
    print(f"  sigma_hat outer / inner   : {outer:.4f} / {inner:.4f} = {outer / inner:.2f}x"
          f"   (true {true_sigma(3.5) / true_sigma(2.0):.2f}x)")
    assert corr > 0.65, corr
    assert outer > 1.6 * inner, (outer, inner)

    # standardized residuals are unit-scale where there is data, which is the
    # statement that the reported uncertainty means what it claims to mean
    z_in = (y_test - mu_test) / sigma_test
    print(f"  std of z in-distribution  : {z_in.std():.4f}")
    assert 0.75 < z_in.std() < 1.35, z_in.std()

    # --- the half that does not exist ---------------------------------------
    #
    # same model, evaluated where it has never seen a point.
    x_gap = np.linspace(GAP[0] + 0.05, GAP[1] - 0.05, 200).astype(np.float32)
    mu_gap, sigma_gap = predict(model, x_gap)
    y_gap_truth = true_mean(x_gap)

    print(f"\nin the gap ({GAP[0]}, {GAP[1]})")
    print(f"  mean |mu - f(x)|          : {np.abs(mu_gap - y_gap_truth).mean():.4f}")
    print(f"  max  |mu - f(x)|          : {np.abs(mu_gap - y_gap_truth).max():.4f}")
    print(f"  mean reported sigma       : {sigma_gap.mean():.4f}")
    print(f"  mean sigma on train range : {predict(model, x_train)[1].mean():.4f}")

    # the mean is wrong in the gap by more than the noise scale there. it has to
    # be - sin(2x) turns over inside the hole and nothing in the data says so.
    assert np.abs(mu_gap - y_gap_truth).max() > 2.0 * true_sigma(x_gap).max()

    # and this is the day. the reported sigma inside the gap is not larger than
    # the reported sigma on the training data - it sits inside the same range, so
    # nothing in the model's output distinguishes "noisy here" from "never been
    # here". stated as a containment rather than as "sigma is small", because a
    # smooth interpolant could produce any value in the middle and only the
    # comparison against the training range is a claim about the model.
    #
    # it comes out *smaller* - 0.12 against 0.32 on the training range - and that
    # is not the model being extra confident out of perversity. 0.12 is very close
    # to the average of true_sigma over the gap, which is where the trend it fitted
    # on both sides extrapolates to. so the head is doing exactly its job and the
    # answer is exactly wrong: it reports the aleatoric term, correctly, in a place
    # where the aleatoric term is not the thing that should dominate.
    sigma_train = predict(model, x_train)[1]
    assert sigma_gap.max() <= sigma_train.max(), (sigma_gap.max(), sigma_train.max())
    assert sigma_gap.mean() < 1.5 * sigma_train.mean()

    # so the bias in units of the model's own claimed width blows up: wrong by a
    # lot, while claiming a width appropriate to being right.
    #
    # this compares against `f(x)` on both sides rather than against a noisy draw,
    # which took two tries to get right. the first version put the in-distribution
    # number on noisy targets and the gap number on the truth, and the gap then
    # looked only 1.2x worse - because the in-distribution side was dominated by
    # the tail of 600 noise draws, an extreme-value statistic, and not by the
    # model's bias at all. two different quantities sharing a name.
    #
    # and it is a *mean* on both sides, for the second half of the same mistake.
    # comparing maxima puts the worst of 600 random draws against the worst of 200
    # grid points and the two are not the same kind of number either; that ratio
    # came out 2.3x while the means differ by an order of magnitude. the maxima are
    # printed anyway because the gap between the two ratios is the point.
    bias_in = np.abs((true_mean(x_test) - mu_test) / sigma_test)
    bias_gap = np.abs((y_gap_truth - mu_gap) / sigma_gap)
    print(f"  mean |bias / sigma| in gap: {bias_gap.mean():.2f}  "
          f"(vs {bias_in.mean():.2f} in-distribution)")
    print(f"  max  |bias / sigma| in gap: {bias_gap.max():.2f}  "
          f"(vs {bias_in.max():.2f} in-distribution)")
    assert bias_gap.mean() > 5.0 * bias_in.mean(), (bias_gap.mean(), bias_in.mean())

    # --- the decomposition, with one model in it -----------------------------
    #
    # not a measurement. the epistemic term is a variance over a point mass and
    # comes out as exactly zero everywhere, in the gap included, which is the
    # structural statement the whole project is a response to. running the real
    # function rather than saying this in a comment, so that the same code path
    # produces the zero today and a real number tomorrow.
    _, aleatoric_one, epistemic_one = decompose([model], x_gap)
    print(f"\nsingle model, in the gap")
    print(f"  aleatoric (mean)          : {aleatoric_one.mean():.5f}")
    print(f"  epistemic (mean)          : {epistemic_one.mean():.5f}")
    assert np.all(epistemic_one == 0.0)

    # --- preview: two initializations ---------------------------------------
    #
    # day 2's object, run here only far enough to establish that the disagreement
    # exists and is localized. the two networks see identical data and differ only
    # in initialization, so anything they disagree about is unconstrained by the
    # data - which is what the epistemic term is supposed to be measuring.
    #
    # what this does NOT establish is that the disagreement is a *calibrated*
    # measure of anything, or that it is not simply larger wherever the network is
    # freer to wobble. that is the question day 2 has to answer and today would
    # answer it wrongly, with n = 2.
    second = train_gaussian_head(x_train, y_train, seed=2)
    _, aleatoric_two, epistemic_two = decompose([model, second], x_gap)
    _, _, epistemic_data = decompose([model, second], x_train)

    print(f"\ntwo initializations")
    print(f"  epistemic in gap  (mean)  : {epistemic_two.mean():.5f}")
    print(f"  epistemic on data (mean)  : {epistemic_data.mean():.5f}")
    print(f"  ratio                     : "
          f"{epistemic_two.mean() / max(epistemic_data.mean(), 1e-12):.1f}x")

    # the disagreement is concentrated in the gap rather than spread over the
    # input range, which is the minimum this has to show to be worth four days
    assert epistemic_two.mean() > 3.0 * epistemic_data.mean()

    # and the aleatoric term is essentially unchanged by adding the second model,
    # which is the sense in which the two terms are separate quantities rather
    # than one quantity split by a convention
    assert np.abs(aleatoric_two.mean() - aleatoric_one.mean()) < 0.5 * aleatoric_one.mean()

    print("\nday 1 checks passed")
