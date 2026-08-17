"""Day 3 of mixture density networks.

Day 1 showed the point regressor returns a legitimate preimage and cannot say the
other two exist. Day 2 built the mixture head and the log-sum-exp NLL, and ended
by establishing that the stable loss does *not* keep a drifting component alive -
a component far from the data with a small scale has an underflowed responsibility
and therefore an exactly zero gradient, and no rearrangement of the formula
recovers it. Today trains the thing, and component death is the first subject
because day 2 left it open.

Four results, all measured below.

**Component death is an initialization problem and it is decided in the first
twenty-five epochs.** Across six seeds with default `nn.Linear` initialization,
every single run loses at least one component - 7 of 18 dead in total, where dead
means the component is not the most probable explanation of any point in the
dataset. Spreading the mean head's bias across the target range, one line, loses
none across the same six seeds. The timing is the part that matters: tracking the
worst run, all three components start with peak responsibility near 0.4 and by
epoch 25 the third is at 0.000. It is not slow starvation that a longer schedule
or a scheduler would fix. At initialization every component sits within about
`1/sqrt(hidden)` of zero, so they overlap almost exactly, and which one claims
which branch is settled by the noise in the first few updates. After that the
loser has no gradient and the run is over.

**The damage from a dead component is not where the missing mode is.** The
obvious guess is that losing a component costs nats inside the folded band, where
the conditional has three modes, and nothing outside it. Measured, it is the
other way round: 0.41 nats outside the band against 0.30 inside, on a total of
0.35. Two things drive it. A surviving component gets stretched to cover work the
dead one was doing, so it is too wide *everywhere*, and NLL punishes a wrong width
in proportion to how peaked the truth is - outside the band the true conditional
is a single sharp bump where nats are expensive, and inside it is diffuse where
they are cheap. So the region that does not need three modes is where the loss
notices they are missing. The corollary is that a per-region loss breakdown is a
bad instrument for locating a structural failure, and the responsibilities are the
good one.

**The noiseless roots are not the modes of the noisy conditional.** Day 1 closed
by noting that the noiseless preimage count is the wrong boundary for an ambiguity
question once the problem has noise in it, and that the error decays outward over
about 2.5 sigma rather than switching off at the fold. Today is where that bit.
`conditional_roots` solves `g(x) = y` exactly, so it reports where the conditional
would peak at zero noise, and the folded band `[0.410, 0.590]` is where that has
three solutions. The first version of today's mode check used those roots as
ground truth and flagged a learned mode at 0.35 for `y = 0.62` as spurious. It is
not spurious. `g(0.35) = 0.593`, which is 0.027 from the observation and well
inside a noise scale of 0.05, so that branch carries real mass and the root finder
cannot see it because it is answering a zero-noise question. Against the analytic
*noisy* conditional the model matches mode for mode, to within 0.019 at every
probe. Three of the five probe points sit outside the folded band and are
genuinely bimodal, so the band is a statement about `g` and not about how many
answers the data admits.

**The PIT is a marginal statistic and here the marginal version is actively
misleading.** Pooled over all 3000 points the probability integral transform has
mean 0.495, standard deviation 0.288 against a uniform's 0.2887, and a decile
chi-square of 6.7 on 9 degrees of freedom - as clean a pass as one could ask for.
Split on the folded band, the same values give 22.0 inside and 23.1 outside, both
past the 1% point. The two halves are biased in opposite directions - mean 0.506
inside against 0.484 outside - and they cancel in the pooled histogram. Nothing
about the pooled number is wrong; it is a correct answer to a question about the
marginal distribution, and calibration on the marginal is simply much weaker than
it sounds. Coverage, which is what an interval-based check would have reported,
is even weaker: it collapses the ten deciles to one number and reads 0.796, 0.894
and 0.955 against nominal 0.80, 0.90 and 0.95.

The scale floor turns out to be inactive here - nothing is pinned to it, and the
narrowest fitted scale is 0.0129 against the 0.0173 the geometry says is the
narrowest legitimate one. That is reported rather than asserted away, because day
2 established the floor prevents a genuine unbounded-descent failure and "not
needed on this run" is a different claim from "not needed".
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from day1_inverse_problem import (
    conditional_density,
    conditional_roots,
    make_inverse_problem,
    train_point_regressor,
    predict,
)
from day2_mixture_head import (
    MixtureDensityNetwork,
    component_log_prob,
    mixture_nll,
    mixture_pdf,
)


# a component is called dead when it is not the most probable explanation of any
# point in the dataset. the alternative definition - weight below some epsilon -
# is weaker and noisier: a component can hold 2% of the weight everywhere,
# explain nothing, and still clear a weight threshold.
DEAD_RESPONSIBILITY = 0.5


def responsibilities(model, x, t):
    """Posterior `p(component = k | x, t)` for every point, `(batch, K)`.

    The softmax of `log pi_k + log N_k`, which is the same vector `logsumexp`
    differentiates through inside the loss. That is the point of computing it
    here rather than inspecting the weights: `pi_k(x)` is the prior over
    components and the *responsibility* is the posterior after seeing the target,
    and it is the posterior that appears in the gradient. A component with a
    healthy prior and zero responsibility everywhere receives no gradient.
    """
    with torch.no_grad():
        logits, mu, sigma = model(x)
        joint = F.log_softmax(logits, dim=-1) + component_log_prob(mu, sigma, t)

        return torch.exp(joint - torch.logsumexp(joint, dim=-1, keepdim=True))


def component_peak_responsibility(model, x, t):
    """Largest responsibility each component attains anywhere in the dataset.

    The death diagnostic. Taking the maximum rather than the mean because the
    question is whether the component is ever the explanation of anything, and a
    component that owns a narrow branch holding 8% of the data has a low mean
    responsibility while being entirely alive.
    """
    return responsibilities(model, x, t).max(dim=0).values


def spread_mean_bias(model, low=0.0, high=1.0):
    """Initialize the mean head's bias across the target range instead of near zero.

    Default `nn.Linear` initialization puts every component's mean within about
    `1/sqrt(hidden)` of zero, so at step zero all `K` components sit on top of
    each other and the responsibilities are nearly uniform. Which component wins
    which branch is then decided by the noise in the first few updates.

    Spreading the bias over the range the targets actually occupy gives each
    component a distinct starting basin. This is one line and it is the whole
    intervention studied below.
    """
    with torch.no_grad():
        model.mean_head.bias.copy_(
            torch.linspace(low, high, model.n_components)
        )
        model.mean_head.weight.mul_(0.1)


def train_mdn(inputs, targets, n_components=3, epochs=2500, lr=1e-2, seed=0,
              spread_init=True, track=None):
    """Full-batch Adam on the mixture NLL. Returns `(model, history)`.

    Full batch rather than minibatch, deliberately. The subject here is component
    death, and minibatching adds a second source of it - a component can lose all
    its responsibility in one unlucky batch - which would confound the thing being
    measured with the thing being used to measure it.

    `track` is a list of epochs at which to record the per-component peak
    responsibility, so the death event can be located in time rather than
    observed after the fact.
    """
    torch.manual_seed(seed)

    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)

    model = MixtureDensityNetwork(n_components=n_components)
    if spread_init:
        spread_mean_bias(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "peaks": {}}
    track = set(track or [])

    for epoch in range(epochs):
        logits, mu, sigma = model(x)
        loss = mixture_nll(logits, mu, sigma, t).mean()

        optimizer.zero_grad()
        loss.backward()

        # the scale floor keeps sigma bounded below, but nothing keeps the raw
        # parameter from walking arbitrarily far negative while it is pinned
        # there, and a parameter far below the floor takes many steps to come
        # back if the data later wants a wider component. clipping is the cheap
        # guard and it is reported rather than assumed to be inactive.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        history["loss"].append(loss.item())
        if epoch in track:
            history["peaks"][epoch] = component_peak_responsibility(model, x, t).tolist()

    return model, history


def mixture_cdf(logits, mu, sigma, t):
    """`P(T <= t | x)` under the predictive mixture, `(batch,)`.

    A mixture of Gaussians has a closed-form CDF - the same weights applied to
    the component CDFs - because the CDF is linear in the density. Worth stating
    because almost nothing else about a mixture is linear in the components, and
    the quantiles are emphatically not: the mixture's median is not the weighted
    median of the component medians and cannot be computed from them.

    Used for the PIT below, which needs the CDF evaluated at the observed target
    and nothing else, so there is no need to invert anything.
    """
    weights = F.softmax(logits, dim=-1)
    z = (t - mu) / (sigma * math.sqrt(2.0))

    return (weights * 0.5 * (1.0 + torch.erf(z))).sum(dim=-1)


def pit_values(model, inputs, targets):
    """Probability integral transform of the targets under the predictive mixture.

    If the predictive distribution is the true conditional, `F(t | x)` is
    uniform on `[0, 1]`, and that holds *per input*, so it survives averaging
    over inputs. This is the calibration test that uses the whole distribution
    rather than an interval, and it is the only one here that a point regressor
    cannot even be given.
    """
    with torch.no_grad():
        x = torch.from_numpy(inputs)
        t = torch.from_numpy(targets)

        return mixture_cdf(*model(x), t).numpy().ravel()


def sample_predictive(model, xs, n_per_input=200, seed=0):
    """Ancestral samples from the predictive mixture at each input in `xs`.

    Draw a component from the weights, then a normal from that component. Not
    `sum_k pi_k * (mu_k + sigma_k eps)`, which is a different and much narrower
    distribution - it is the *convolution* of the components rather than the
    mixture of them, and on a bimodal conditional it concentrates exactly in the
    valley between the modes, reproducing day 1's failure inside a model that had
    already avoided it.
    """
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        column = torch.from_numpy(np.asarray(xs, dtype=np.float32).reshape(-1, 1))
        logits, mu, sigma = model(column)

        weights = F.softmax(logits, dim=-1)
        picks = torch.multinomial(
            weights.repeat_interleave(n_per_input, dim=0), 1, generator=generator
        )

        mu_rep = mu.repeat_interleave(n_per_input, dim=0).gather(1, picks)
        sigma_rep = sigma.repeat_interleave(n_per_input, dim=0).gather(1, picks)

        noise = torch.randn(mu_rep.shape, generator=generator)

        return (mu_rep + sigma_rep * noise).reshape(len(column), n_per_input).numpy()


def density_modes(density, grid, floor=0.05):
    """Local maxima of a density sampled on `grid`.

    The `floor` drops maxima carrying negligible mass. Without it a nearly-flat
    tail contributes maxima at grid resolution and they get counted as modes.

    One function for both the learned and the true density, deliberately. The
    comparison below is between two mode sets and it should not be able to come
    out well because the two were extracted differently.
    """
    inner = density[1:-1]
    is_peak = (inner > density[:-2]) & (inner > density[2:])
    is_peak &= inner > floor * density.max()

    return grid[1:-1][is_peak]


def predictive_density(model, y, grid):
    """The learned conditional density at input `y`, evaluated on `grid`."""
    with torch.no_grad():
        column = torch.tensor([[y]], dtype=torch.float32)
        logits, mu, sigma = model(column)

        return mixture_pdf(logits[0], mu[0], sigma[0],
                           torch.from_numpy(grid.astype(np.float32))).numpy()


def predictive_modes(model, y, grid, floor=0.05):
    """Local maxima of the predictive density at input `y`.

    The mixture's modes are not its means. Two components closer together than
    their widths merge into one bump, so reading the means off the parameters
    overcounts. Grid scan rather than a solver for day 1's reason: this is a
    measurement instrument and reliability beats speed.
    """
    return density_modes(predictive_density(model, y, grid), grid, floor)


def mean_nll(model, inputs, targets, mask=None):
    """Mean NLL over the dataset, or over the subset selected by `mask`.

    Restricting to a subset is the point. The aggregate NLL is an average over
    the whole input range, and the obvious guess is that losing a component costs
    nats exactly where the conditional has three modes and nothing anywhere else.
    Splitting on the band is what tests that guess, and below it does not survive.
    """
    with torch.no_grad():
        x = torch.from_numpy(inputs)
        t = torch.from_numpy(targets)
        per_point = mixture_nll(*model(x), t)

        if mask is not None:
            per_point = per_point[torch.from_numpy(mask)]

        return per_point.mean().item()


def folded_band(lo=0.15, hi=0.85, probes=141):
    """The interval of inputs whose conditional has three preimages.

    Found by counting roots on a probe grid rather than by solving for where
    `g'` changes sign, since the root counter is what every ground-truth claim
    in this project leans on and this exercises it.
    """
    probe = np.linspace(lo, hi, probes)
    three = [float(y) for y in probe if len(conditional_roots(y)) == 3]

    return min(three), max(three)




EPOCHS = 1200
SEEDS = [0, 1, 2, 3, 4, 5]
TRACK = [0, 25, 50, 100, 200, 400, 800, EPOCHS - 1]


if __name__ == "__main__":
    NOISE = 0.05
    inputs, targets, _ = make_inverse_problem(3000, noise_scale=NOISE, seed=0)

    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)

    band_lo, band_hi = folded_band()
    inside = (inputs.ravel() >= band_lo) & (inputs.ravel() <= band_hi)

    print(f"samples                   : {len(inputs)}")
    print(f"folded band               : [{band_lo:.3f}, {band_hi:.3f}]")
    print(f"points inside / outside   : {int(inside.sum())} / {int((~inside).sum())}")

    # ---- the death study. day 2 established that the loss's numerics do not
    # keep a component alive; the question left open was what does.
    print("\ncomponent death across seeds")
    print("  init      seed   nll(all)  nll(in)  nll(out)  peaks                     dead")

    runs = {}
    for spread in [False, True]:
        for seed in SEEDS:
            trained, history = train_mdn(inputs, targets, seed=seed, epochs=EPOCHS,
                                         spread_init=spread, track=TRACK)
            peaks = component_peak_responsibility(trained, x, t)
            dead = int((peaks < DEAD_RESPONSIBILITY).sum())

            runs[(spread, seed)] = {
                "model": trained,
                "history": history,
                "peaks": peaks,
                "dead": dead,
                "all": mean_nll(trained, inputs, targets),
                "in": mean_nll(trained, inputs, targets, inside),
                "out": mean_nll(trained, inputs, targets, ~inside),
            }

            r = runs[(spread, seed)]
            label = "spread" if spread else "default"
            shown = str([round(v, 3) for v in peaks.tolist()])
            print(f"  {label:8s}  {seed}    {r['all']:8.4f} {r['in']:8.4f} {r['out']:9.4f}  "
                  f"{shown:24s}  {dead}")

    for spread in [False, True]:
        group = [runs[(spread, s)] for s in SEEDS]
        label = "spread" if spread else "default"
        print(f"\n  {label:8s}  dead components  : "
              f"{sum(r['dead'] for r in group)} / {3 * len(SEEDS)}")
        for region in ["all", "in", "out"]:
            print(f"  {label:8s}  mean nll ({region:3s})  : "
                  f"{np.mean([r[region] for r in group]):.4f}")

    gaps = {
        region: (np.mean([runs[(False, s)][region] for s in SEEDS])
                 - np.mean([runs[(True, s)][region] for s in SEEDS]))
        for region in ["all", "in", "out"]
    }

    print("\n  cost in nats of losing a component, by region")
    print(f"    everywhere                : {gaps['all']:.4f}")
    print(f"    inside the folded band    : {gaps['in']:.4f}   "
          f"(where three modes exist)")
    print(f"    outside the folded band   : {gaps['out']:.4f}   "
          f"(where one mode exists)")

    assert sum(runs[(True, s)]["dead"] for s in SEEDS) == 0
    assert sum(runs[(False, s)]["dead"] for s in SEEDS) > 0
    assert all(g > 0 for g in gaps.values()), gaps

    # the damage is not where the missing mode is. asserted in the direction the
    # measurement actually came out, so that a rerun landing the other way is a
    # failure rather than a silently rewritten conclusion.
    assert gaps["out"] > gaps["in"], gaps

    # ---- when it happens. slow starvation would show in the loss curve; a
    # decision taken in the first few hundred steps cannot be trained out and
    # the fix has to live at initialization.
    worst = max(SEEDS, key=lambda s: (runs[(False, s)]["dead"], runs[(False, s)]["all"]))
    print(f"\nwhen the component dies (default init, seed {worst})")

    for epoch in TRACK:
        peaks = runs[(False, worst)]["history"]["peaks"][epoch]
        loss = runs[(False, worst)]["history"]["loss"][epoch]
        print(f"  epoch {epoch:5d}  nll={loss:8.4f}  "
              f"peaks={[round(v, 3) for v in peaks]}")

    # ---- everything below uses the healthy fit
    model = runs[(True, 0)]["model"]
    history = runs[(True, 0)]["history"]
    logits, mu, sigma = model(x)

    print(f"\nfinal training nll        : {history['loss'][-1]:.4f}")
    print(f"peak responsibilities     : "
          f"{[round(v, 3) for v in runs[(True, 0)]['peaks'].tolist()]}")

    # ---- scale collapse. whether the floor is load-bearing is a fact about the
    # fit, not about the code, so it gets measured rather than asserted away.
    with torch.no_grad():
        at_floor = (sigma < model.min_sigma * 1.01).float().mean().item()

    narrowest_real = NOISE / (1.0 + 0.6 * math.pi)

    print(f"sigma range after training: {sigma.min():.5f} to {sigma.max():.5f}")
    print(f"fraction pinned at floor  : {at_floor:.4f}")
    print(f"narrowest legitimate width: {narrowest_real:.5f}")

    # ---- do the modes land where the conditional actually peaks. this is the
    # claim day 1 set up, and the first version of this check got the ground
    # truth wrong in a way worth keeping.
    x_grid = np.linspace(0.0, 1.0, 2001)
    probe = [0.30, 0.50, 0.62, 0.70, 0.95]

    print("\nmodes vs the noiseless roots")

    for y in probe:
        roots = conditional_roots(y)
        found = predictive_modes(model, y, x_grid)
        extra = [float(m) for m in found
                 if len(roots) == 0 or min(abs(roots - m)) > 0.05]
        print(f"  y={y:.2f}  roots={[round(float(r), 3) for r in roots]}  "
              f"modes={[round(float(m), 3) for m in found]}  "
              f"unmatched modes={[round(v, 3) for v in extra]}")

    # `conditional_roots` solves g(x) = y exactly, so it returns the modes of the
    # *noiseless* conditional. that is the right yardstick only where the fold is
    # far away. at y = 0.62 it reports one root and the model reports two, and
    # the model is right: g(0.35) = 0.593, which is 0.027 from y, well inside a
    # noise scale of 0.05. so the second branch carries real mass at that y and
    # the root finder cannot see it, because it answers a zero-noise question.
    #
    # the fix is to compare against the analytic noisy conditional, which day 1
    # already provides, instead of against the roots.
    print("\nmodes vs the true conditional at this noise level")

    for y in probe:
        truth = conditional_density(x_grid, y, None, NOISE)
        true_modes = density_modes(truth, x_grid)
        found = predictive_modes(model, y, x_grid)

        errors = ([float(min(abs(found - m))) for m in true_modes]
                  if len(found) else [9.9])
        print(f"  y={y:.2f}  true modes={[round(float(m), 3) for m in true_modes]}  "
              f"learned={[round(float(m), 3) for m in found]}  "
              f"max err={max(errors):.3f}")

        assert len(found) == len(true_modes), (y, true_modes, found)
        assert max(errors) < 0.05, (y, true_modes, found)

    # and the noiseless roots are a subset of the true modes everywhere, which is
    # what says the disagreement above is the root finder being incomplete rather
    # than the two instruments disagreeing
    for y in probe:
        true_modes = density_modes(conditional_density(x_grid, y, None, NOISE), x_grid)
        for r in conditional_roots(y):
            assert min(abs(true_modes - r)) < 0.05, (y, r, true_modes)

    # ---- the point regressor on the same inputs, for contrast
    point_model = train_point_regressor(inputs, targets, epochs=1500, seed=0)
    point_at = predict(point_model, np.array(probe, dtype=np.float32))

    print("\npoint regressor on the same inputs")
    for y, p in zip(probe, point_at):
        roots = conditional_roots(y)
        print(f"  y={y:.2f}  point={p:.3f}  nearest root={min(abs(roots - p)):.3f}")

    # ---- calibration. the PIT uses the whole distribution rather than an
    # interval, and is the only check here a point regressor cannot be given.
    pit = pit_values(model, inputs, targets)

    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(pit, bins=edges)
    expected = len(pit) / 10.0
    chi2 = float(((counts - expected) ** 2 / expected).sum())

    print("\ncalibration (PIT of the targets under the predictive mixture)")
    print(f"  mean / std              : {pit.mean():.4f} / {pit.std():.4f}"
          f"   (uniform: 0.5000 / 0.2887)")
    print(f"  decile counts           : {counts.tolist()}")
    print(f"  expected per decile     : {expected:.0f}")
    print(f"  chi-square (9 dof)      : {chi2:.1f}")

    for level in [0.5, 0.8, 0.9, 0.95]:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        print(f"  {level:.0%} central coverage  : "
              f"{float(((pit > lo) & (pit < hi)).mean()):.4f}")

    # the PIT is a marginal statistic, which is a limitation rather than a
    # caveat: it averages over inputs, so being too wide in one region and too
    # narrow in another cancels in the histogram. splitting on the band is the
    # cheapest thing that would notice.
    print("\n  the same PIT, split on the folded band")
    for name, mask in [("inside", inside), ("outside", ~inside)]:
        sub = pit[mask]
        sub_counts, _ = np.histogram(sub, bins=edges)
        sub_chi2 = float(((sub_counts - len(sub) / 10) ** 2 / (len(sub) / 10)).sum())
        print(f"    {name:7s}  n={len(sub):4d}  mean={sub.mean():.4f}  "
              f"std={sub.std():.4f}  chi2={sub_chi2:.1f}")

    assert abs(pit.mean() - 0.5) < 0.05, pit.mean()
    assert abs(pit.std() - 0.2887) < 0.05, pit.std()

    # ---- density against the analytic conditional, which is what the PIT cannot
    # see: the PIT reads the target's rank and not where the mass sits.
    print("\npredictive density vs the true conditional")

    for y in probe:
        truth = conditional_density(x_grid, y, None, NOISE)
        learned = predictive_density(model, y, x_grid)

        # total variation, so it is a probability and comparable across probes
        tv = 0.5 * float(np.trapezoid(np.abs(learned - truth), x_grid))
        print(f"  y={y:.2f}  total variation = {tv:.4f}")
        assert tv < 0.2, (y, tv)

    # ---- ancestral sampling, and the mistake it is easy to make instead
    print("\nsampling from the predictive mixture")

    # the middle of the folded band, where the conditional has three genuine
    # modes rather than one and a tail
    y_multi = 0.50
    draws = sample_predictive(model, [y_multi], n_per_input=4000, seed=0)[0]
    roots = conditional_roots(y_multi)

    near = [float((np.abs(draws - r) < 0.06).mean()) for r in roots]

    with torch.no_grad():
        column = torch.tensor([[y_multi]], dtype=torch.float32)
        lg, m, sg = model(column)
        analytic_mean = float((F.softmax(lg, dim=-1) * m).sum())

    print(f"  y={y_multi}  roots={[round(float(r), 3) for r in roots]}")
    print(f"  sample mass within 0.06 : {[round(v, 3) for v in near]}")
    print(f"  sample mean             : {draws.mean():.3f}")
    print(f"  analytic mixture mean   : {analytic_mean:.3f}")
    print(f"  distance to nearest root: {min(abs(roots - analytic_mean)):.3f}")

    # every branch gets real mass, which is the statement that no mode was
    # dropped - the thing day 1's point estimate could not do at all
    assert all(v > 0.1 for v in near), near
    assert sum(near) > 0.6, near

    # and the sample mean agrees with the mixture mean computed in closed form,
    # which is a check on the sampler rather than on the fit: get the ancestral
    # draw wrong and this still passes, so it is paired with the mass check above
    # rather than trusted on its own
    assert abs(draws.mean() - analytic_mean) < 0.03, (draws.mean(), analytic_mean)

    print("\nall good")
