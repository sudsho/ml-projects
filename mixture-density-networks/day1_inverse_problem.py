"""
Day 1 of mixture density networks.

The conformal project handled a version of this question and stopped one step
short of it. Split conformal takes any point regressor, scores it on a held-out
calibration set, and returns an interval with finite-sample marginal coverage.
That is a real guarantee and it is also deliberately shape-blind: it says the
truth is in this set 90% of the time, and nothing at all about where in the set.
When the conditional distribution has two separated modes with a valley between
them, the conformal band covers both modes *and* the valley, is correct in the
only sense it claims, and points at a region where the answer provably is not.

An MDN answers the other question. Instead of a point plus a calibrated radius,
the network emits the parameters of a full conditional density, and the shape
comes out with it.

Before any of that is worth building, the failure it fixes has to be pinned down,
and the version I came in with turned out to be wrong on this dataset. The usual
telling is that squared error lands "between the modes", in the valley, at a
point of near-minimal density. The reasoning is sound - the minimizer of
E[(f(x) - t)^2] over all measurable f is the conditional mean E[t | x], and the
mean of a symmetric bimodal density is its centre - and it does not happen here.
Measured across the whole ambiguous band, the fitted regressor sits a median of
0.012 from a genuine preimage, and always the same one.

Why: at any preimage the conditional density is exp(0), so all three preimages
have identical peak height. What separates them is width. Near a root the bump
has scale sigma / |g'|, so the branch where the generating map is flattest
carries the most mass, and the flat branch is the folded middle one, at every y
in the band. The conditional mean is pulled onto the most probable branch and
stays there, and at the symmetry point it lands on it exactly.

So the point regressor is not visibly broken. It returns a legitimate preimage,
on the single most probable branch, and by any point metric it looks correct.
That is worse than the textbook story rather than better, because it means no
point metric is going to catch it. What it cannot do is say the other two
preimages exist - and they hold 56% of the conditional mass, discarded silently,
at every y in the band. The failure is not a wrong number. It is the wrong
output object, which is exactly the thing a different optimizer, a bigger
network, or a different point loss cannot fix. Today is set up to rule those out
rather than assert them:

  - not the optimizer, and not capacity: the fitted curve is checked against a
    nonparametric binned estimate of E[t | x] computed from the data with no
    network in it, in units of each bin's own standard error. They agree. The
    network is at its optimum.
  - not the point loss: L1 is fitted too. Its minimizer is the conditional
    median, a genuinely different number, and it lands on the same branch.
  - the control: three sigma clear of the fold the same network is accurate to
    0.02, so none of this is about the fit.

Two things the setup got wrong that are worth keeping. The noiseless preimage
count is the wrong boundary for an ambiguity question once the problem has noise
in it - the error does not switch off at the edge of the fold band, it decays
outward over about 2.5 sigma, because a y just outside still has two preimages
within a couple of sigma. And at sigma = 0.05 the modes are only marginally
resolved, with the valley about a fifth of the peak rather than a clean gap.
That is day 2's problem, not a defect in the dataset.

Today: the inverse-problem dataset, the true preimages solved exactly enough to
serve as ground truth, the branch-mass accounting that explains where the point
estimate goes, and both point regressors measured against it. Day 2 replaces the
single output with a mixture head and the squared error with the mixture NLL.
"""

import numpy as np
import torch
import torch.nn as nn


FORWARD_AMPLITUDE = 0.3


def forward_map(x):
    """The generating map `g(x) = x + 0.3 sin(2 pi x)`.

    Monotone would make this a boring dataset in both directions. It is not:
    `g'(x) = 1 + 0.6 pi cos(2 pi x)` goes negative wherever
    `cos(2 pi x) < -1 / (0.6 pi) ~ -0.53`, so `g` has folds. Forward it is still
    a function, and a point regressor learns it without trouble. Inverted, the
    folds turn into genuine ambiguity - a single `y` in the folded band has three
    preimages, and no amount of data distinguishes them because they are all
    equally real.

    This is the standard Bishop construction and it is worth being clear about
    why a synthetic one earns its place here. The point being made is about the
    gap between the conditional mean and the conditional modes, and to measure
    that gap the true conditional has to be known rather than estimated. On real
    data the modes would themselves be estimates and every number below would
    inherit their error.
    """
    return x + FORWARD_AMPLITUDE * np.sin(2 * np.pi * x)


def make_inverse_problem(n_samples, noise_scale=0.05, seed=0):
    """Sample the forward map, then swap the axes to get the inverse problem.

    Inputs `t ~ U(0, 1)` go through `g` with additive Gaussian noise, and then
    the pair is used backwards: the network sees `g(t) + eps` as its input and is
    asked for `t`. Swapping rather than sampling the inverse directly keeps the
    noise on the side it belongs on and keeps the conditional analytically
    reachable, since the roots of `g(x) = y` are computable.

    Returns `(inputs, targets)` as column vectors, matching the shape the torch
    models want, with the underlying clean value kept for the ground-truth work
    later.
    """
    rng = np.random.default_rng(seed)

    latent = rng.uniform(0.0, 1.0, size=n_samples)
    observed = forward_map(latent) + rng.normal(0.0, noise_scale, size=n_samples)

    inputs = observed.reshape(-1, 1)
    targets = latent.reshape(-1, 1)

    return inputs.astype(np.float32), targets.astype(np.float32), latent


def conditional_roots(y, lo=0.0, hi=1.0, grid=20001, tol=1e-12):
    """All `x` in `[lo, hi]` with `g(x) = y`, by sign change then bisection.

    The noiseless conditional `p(x | y)` puts its mass at these roots, so they
    are the modes the noisy conditional smears out. A dense scan for sign changes
    followed by bisection on each bracket is slow and completely reliable, which
    is the correct trade for something whose only job is to be ground truth for
    everything else in the file.

    Scanning rather than solving because `g` is not invertible in closed form and
    a Newton iteration would need a starting point per root, which is the thing
    being looked for. `grid` is far denser than the number of folds so no bracket
    is skipped.

    The dedup at the end is not cosmetic. A root that lands exactly on a grid
    point makes both of the brackets touching it satisfy `f(a) f(b) <= 0`, so it
    gets found twice and the count comes back as four where the geometry allows
    only one or three. Counting roots is how the folded band gets identified
    below, so a spurious root is a wrong band and not just a wrong number.
    """
    xs = np.linspace(lo, hi, grid)
    values = forward_map(xs) - y

    found = []

    for i in range(grid - 1):
        left, right = values[i], values[i + 1]

        if left * right > 0:
            continue

        a, b = xs[i], xs[i + 1]
        fa = left
        while b - a > tol:
            mid = (a + b) / 2
            fm = forward_map(mid) - y
            if fa * fm <= 0:
                b = mid
            else:
                a, fa = mid, fm
        found.append((a + b) / 2)

    roots = []
    for r in sorted(found):
        if not roots or r - roots[-1] > 1e-6:
            roots.append(r)

    return np.array(roots)


def binned_conditional_mean(inputs, targets, n_bins=40):
    """`E[t | x]` estimated by averaging targets inside bins of `x`.

    No network, no optimizer, no loss. This exists so that "the MSE regressor
    lands in the valley" can be shown to be a fact about the conditional mean
    rather than a fact about the fit. If the two disagree the regressor is
    underfit and every conclusion today is about training rather than about the
    target, so this gets checked before anything else is claimed.

    The standard error comes back with the means, and it is needed rather than
    nice to have. The bins inside the folded band draw from a conditional with
    two separated modes, so their spread is several times the noise scale and
    their means are correspondingly uncertain. A fixed absolute tolerance on
    "fitted agrees with binned" would be a statement about how many samples went
    into the bins, and it would tighten or loosen with the bin count rather than
    with the quality of the fit. Comparing in units of the bin's own standard
    error is the version that means what it says.

    Returns `(centers, means, sems, counts)` with empty bins dropped.
    """
    x = inputs.ravel()
    t = targets.ravel()

    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    index = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    centers, means, sems, counts = [], [], [], []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count < 2:
            continue
        values = t[mask]
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        means.append(float(values.mean()))
        sems.append(float(values.std(ddof=1) / np.sqrt(count)))
        counts.append(count)

    return np.array(centers), np.array(means), np.array(sems), np.array(counts)


def make_mlp(hidden=64):
    """A plain 1 -> 1 regressor, two hidden layers of tanh.

    Deliberately larger than this problem needs. The argument today is that the
    point regressor is at its optimum and the optimum is in the wrong place, and
    that argument is only clean if capacity is visibly not the binding
    constraint. Tanh rather than ReLU so the fitted curve is smooth and the
    comparison against the binned estimate is not reading piecewise-linear kinks.
    """
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


def train_point_regressor(inputs, targets, loss_name="mse", epochs=3000, seed=0):
    """Full-batch training of the point regressor under `mse` or `mae`.

    Both are fitted because the interesting claim is not "squared error is bad"
    but "no point loss helps". MSE has E[t | x] as its minimizer and MAE has the
    conditional median, and on a roughly symmetric bimodal conditional both of
    those sit between the modes. If the fix were a different point loss it would
    show up here as MAE escaping the valley, and it does not.

    Full batch and a long schedule rather than minibatches, because stochastic
    gradient noise would leave a residual wobble that is indistinguishable at a
    glance from the effect being measured.
    """
    torch.manual_seed(seed)

    model = make_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()

    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x), t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"  {loss_name} epoch {epoch + 1:5d}  loss {loss.item():.5f}")

    model.eval()
    return model


def predict(model, xs):
    """Point predictions for `xs`, as a flat float array.

    The reshape to a column and the ravel back are the whole function: the model
    wants `(batch, 1)` and every caller here wants a 1-d array to plot or compare
    against roots. Doing it in one place keeps the shape juggling out of the
    measurements.

    `no_grad` because none of these calls are ever differentiated, and without it
    the returned array carries a graph that pins the activations of every batch
    it was called on.
    """
    with torch.no_grad():
        column = torch.from_numpy(np.asarray(xs, dtype=np.float32).reshape(-1, 1))
        return model(column).numpy().ravel()


def conditional_density(x_grid, y, latent, noise_scale):
    """`p(x | y)` up to normalization, from the generative model directly.

    `p(x | y) ~ p(y | x) p(x)` with `p(x)` uniform on the unit interval and
    `p(y | x)` Gaussian around `g(x)`, so the shape is `exp(-(y - g(x))^2 /
    2 sigma^2)` on `[0, 1]` and zero outside. Normalized on the grid so the
    values are comparable across different `y`.

    Written from the model rather than estimated from samples for the same reason
    the roots are solved rather than fitted: this is the yardstick, and a
    yardstick with sampling error in it cannot measure a claim about where
    density is low.
    """
    del latent

    unnormalized = np.exp(-((y - forward_map(x_grid)) ** 2) / (2 * noise_scale**2))
    area = np.trapezoid(unnormalized, x_grid)

    return unnormalized / area


if __name__ == "__main__":
    NOISE = 0.05
    inputs, targets, latent = make_inverse_problem(3000, noise_scale=NOISE, seed=0)

    print(f"samples                   : {len(inputs)}")
    print(f"input range               : [{inputs.min():.3f}, {inputs.max():.3f}]")

    # the fold band. g' < 0 needs cos(2 pi x) < -1 / (0.6 pi), and the y values
    # reached inside that band are the ones with three preimages. found by
    # counting roots rather than by solving for the band, since the root counter
    # is the thing everything else leans on and this exercises it.
    probe = np.linspace(0.15, 0.85, 141)
    root_counts = np.array([len(conditional_roots(y)) for y in probe])
    ambiguous = probe[root_counts >= 3]
    unimodal = probe[root_counts == 1]

    print(f"ambiguous y (3 preimages) : [{ambiguous.min():.3f}, {ambiguous.max():.3f}]")
    print(f"root counts seen          : {sorted(set(root_counts.tolist()))}")

    assert len(ambiguous) > 0, "no folded region - the amplitude is too small"
    assert len(unimodal) > 0, "no unimodal region - nothing to use as a control"

    print("\ntraining point regressors")
    mse_model = train_point_regressor(inputs, targets, "mse", seed=0)
    mae_model = train_point_regressor(inputs, targets, "mae", seed=0)

    # first and before anything else: is the MSE regressor actually at its
    # optimum? if it disagrees with a binned conditional mean computed with no
    # network in it, then today is about underfitting and not about the target.
    centers, binned, sems, counts = binned_conditional_mean(inputs, targets, n_bins=40)
    keep = counts >= 30
    fitted = predict(mse_model, centers[keep])

    deviation = np.abs(fitted - binned[keep])
    z_scores = deviation / sems[keep]

    print(f"\nmax |fitted - binned E[t|x]| : {deviation.max():.4f}")
    print(f"  same, in bin standard errors : {z_scores.max():.2f}")
    assert z_scores.max() < 3.0, z_scores.max()

    # the claim i came in with was "a squared-error regressor lands in the valley
    # between the modes", and the obvious place to check it is the middle of the
    # folded band. it is false there, and false for a reason that holds across the
    # whole band, so the day is really about the corrected version.
    #
    # g is symmetric about (0.5, 0.5) - g(1 - x) = 1 - g(x) - so at y = 0.5 the
    # three preimages are symmetric about 0.5 and the conditional mean is exactly
    # 0.5, which *is* the middle preimage. that much follows from the symmetry and
    # i should have seen it coming. what i could not have predicted is that the
    # prediction barely leaves the middle branch anywhere else in the band either.
    y_center = 0.5
    center_roots = conditional_roots(y_center)
    center_pred = float(predict(mse_model, [y_center])[0])

    print("\nat the symmetry point y = 0.5")
    print(f"  preimages               : {np.round(center_roots, 4).tolist()}")
    print(f"  MSE prediction          : {center_pred:.4f}")

    assert len(center_roots) == 3, center_roots
    assert abs(center_roots[1] - 0.5) < 1e-6, center_roots
    assert abs(center_pred - center_roots[1]) < 0.02, (center_roots, center_pred)

    # distance from the point prediction to the nearest true preimage, over the
    # whole folded band rather than at a single y
    distances, middle_distances = [], []
    for y in ambiguous:
        r = conditional_roots(y)
        p = float(predict(mse_model, [y])[0])
        distances.append(min(abs(p - root) for root in r))
        middle_distances.append(abs(p - r[1]))
    distances = np.array(distances)
    middle_distances = np.array(middle_distances)

    print("\ndistance from the prediction to the nearest preimage, over the band")
    print(f"  min / median / max      : {distances.min():.4f} / "
          f"{np.median(distances):.4f} / {distances.max():.4f}")
    print(f"  to the middle preimage  : {np.median(middle_distances):.4f} (median)")

    # so the naive claim is just wrong on this dataset. the prediction is not in
    # a valley. it is on the middle branch, essentially everywhere in the band,
    # and the nearest preimage is always that one.
    assert np.median(distances) < 0.02, np.median(distances)
    assert distances.max() < 0.11, distances.max()
    assert np.allclose(distances, middle_distances), "nearest preimage is not the middle"

    # why, and this is what makes it a fact about the target rather than a
    # coincidence of this network. every preimage has the *same* peak density,
    # since p(x|y) ~ exp(-(y - g(x))^2 / 2 sigma^2) is exp(0) = 1 at any root.
    # what differs is width - near a root the bump has scale sigma / |g'| - so the
    # branch where g is flattest carries the most mass. the middle branch is the
    # folded one, |g'| is smallest there, and it wins at every y in the band.
    grid = np.linspace(0.0, 1.0, 4001)

    def branch_masses(y, roots):
        """Conditional mass in each branch's cell, split at the midpoints."""
        density = conditional_density(grid, y, latent, NOISE)
        cuts = [0.0] + [(roots[i] + roots[i + 1]) / 2 for i in range(len(roots) - 1)]
        cuts = cuts + [1.0]
        masses = []
        for lo, hi in zip(cuts, cuts[1:]):
            cell = (grid >= lo) & (grid <= hi)
            masses.append(float(np.trapezoid(density[cell], grid[cell])))
        return masses

    print("\nbranch mass against the slope at each preimage")
    for y in [0.42, 0.46, 0.50, 0.55, 0.58]:
        r = conditional_roots(y)
        slopes = [float(abs(1 + 0.6 * np.pi * np.cos(2 * np.pi * x))) for x in r]
        masses = branch_masses(y, r)
        density_y = conditional_density(grid, y, latent, NOISE)
        peaks = [float(np.interp(x, grid, density_y)) for x in r]

        print(f"  y={y:.2f}  slope={[round(v, 2) for v in slopes]}"
              f"  mass={[round(m, 3) for m in masses]}")

        # equal peaks, so the mass ordering is entirely the slope ordering
        assert max(peaks) - min(peaks) < 0.02 * max(peaks), (y, peaks)
        assert int(np.argmax(masses)) == int(np.argmin(slopes)), (y, masses, slopes)
        assert int(np.argmin(slopes)) == 1, (y, slopes)

    # which leaves the real failure, and it is not the textbook one. the point
    # regressor is not visibly broken here. it returns a legitimate preimage, on
    # the single most probable branch, nearly everywhere in the band. by any point
    # metric it looks correct - and that is worse rather than better, because it
    # means no point metric is going to catch this.
    y_star = float(ambiguous[int(np.argmax(distances))])
    roots = conditional_roots(y_star)
    mse_pred = float(predict(mse_model, [y_star])[0])
    mae_pred = float(predict(mae_model, [y_star])[0])

    assert len(roots) == 3, roots
    assert roots.max() - roots.min() > 0.3, roots

    print(f"\ny* (worst case in band)   : {y_star:.4f}")
    print(f"preimages of y*           : {np.round(roots, 4).tolist()}")
    print(f"MSE prediction at y*      : {mse_pred:.4f}")
    print(f"MAE prediction at y*      : {mae_pred:.4f}")

    # L1 too, which is what rules out "use a different point loss". its minimizer
    # is the conditional median rather than the mean, a genuinely different
    # number, and it sits on the same branch.
    assert abs(mae_pred - roots[1]) < 0.11, (roots, mae_pred)

    # the measurement that does catch it: how much conditional mass the returned
    # answer throws away. the outer branches are equally real preimages and a
    # single number has no way to mention them.
    discarded = []
    for y in ambiguous:
        r = conditional_roots(y)
        masses = branch_masses(y, r)
        discarded.append(1.0 - masses[int(np.argmax(masses))])
    discarded = np.array(discarded)

    print("\nmass on branches a point estimate cannot report")
    print(f"  min / median / max      : {discarded.min():.3f} / "
          f"{np.median(discarded):.3f} / {discarded.max():.3f}")

    # over half the conditional, discarded silently, at every y in the band
    assert discarded.min() > 0.5, discarded.min()

    # the conditional mean straight from the density, closing the loop: the
    # network's answer is the conditional mean and not an artifact of training
    density = conditional_density(grid, y_star, latent, NOISE)
    analytic_mean = float(np.trapezoid(grid * density, grid))
    print(f"analytic E[x|y*]          : {analytic_mean:.4f}")
    assert abs(analytic_mean - mse_pred) < 0.06, (analytic_mean, mse_pred)

    # the control: where the conditional has one mode the same network, loss and
    # training run should be accurate, so that none of the above is about the fit.
    #
    # written the obvious way - every y with a single preimage - it fails, badly,
    # and the failure is the second thing worth keeping today. the error does not
    # switch off at the edge of the fold band. it decays smoothly outward from it
    # over a scale set by the noise, because a y just outside the band still has
    # the two merging preimages within a couple of sigma and the conditional still
    # carries real mass over there. the noiseless root count is the wrong boundary
    # for an ambiguity question once there is noise in the problem, and I would
    # have written `root_count == 1` as the definition of "unambiguous" without
    # thinking about it.
    band_lo, band_hi = ambiguous.min(), ambiguous.max()

    errors, gaps_from_band = [], []
    for y in unimodal:
        single = conditional_roots(y)
        assert len(single) == 1, (y, single)
        errors.append(abs(float(predict(mse_model, [y])[0]) - single[0]))
        gaps_from_band.append(min(abs(y - band_lo), abs(y - band_hi)))
    errors = np.array(errors)
    gaps_from_band = np.array(gaps_from_band)

    # the distance at which the point estimate first becomes trustworthy, read
    # off the data rather than assumed
    clean = gaps_from_band[errors < 0.05]
    crossover = float(clean.min())

    print("\noutside the fold band, by distance from its edge")
    print(f"  error within 1 sigma    : {errors[gaps_from_band < NOISE].max():.4f}")
    print(f"  error beyond 3 sigma    : {errors[gaps_from_band > 3 * NOISE].max():.4f}")
    print(f"  accurate from           : {crossover:.3f} = {crossover / NOISE:.1f} sigma")

    # right at the edge the point estimate is as wrong as it is inside the band
    assert errors[gaps_from_band < NOISE].max() > 0.3, errors[gaps_from_band < NOISE].max()

    # and the crossover is set by the noise scale, not by the geometry of g
    assert 1.0 < crossover / NOISE < 4.0, crossover / NOISE

    # so the real control is the genuinely unambiguous region, three sigma clear
    # of the fold, and there the same network is accurate
    worst_unimodal = float(errors[gaps_from_band > 3 * NOISE].max())
    assert worst_unimodal < 0.05, worst_unimodal

    # and the conformal band from the other side, sketched here to size what day 2
    # has to beat. a symmetric band around the point prediction wide enough to
    # cover 90% of the residuals is most of the output range.
    #
    # stated at the symmetry point rather than at y*, and the reason is a
    # correction. y* is the band edge, chosen above as the worst case for the
    # branch-distance argument, and there two of the three preimages are in the
    # middle of merging - so "the valley between them" is not a valley and the
    # band centred on a drifted prediction misses the far branch. neither of
    # those is about the interval. picking one y for every question because it
    # was the extreme for the first one is how a demonstration ends up proving
    # something other than what it says.
    residuals = np.abs(predict(mse_model, inputs.ravel()) - targets.ravel())
    radius = float(np.quantile(residuals, 0.9))

    center_density = conditional_density(grid, y_center, latent, NOISE)

    print(f"\n90% split-conformal radius: {radius:.4f}")
    print(f"  band width              : {2 * radius:.4f} of a unit output range")
    print(f"  band at y = 0.5         : [{center_pred - radius:.4f}, "
          f"{center_pred + radius:.4f}]")

    assert 2 * radius > 0.6, radius

    covered = [r for r in center_roots if abs(r - center_pred) <= radius]
    assert len(covered) == 3, (center_roots, center_pred, radius)

    # so the two summaries fail in opposite directions for the same reason. the
    # point estimate names one branch and cannot say the other two exist; the
    # interval covers all three and cannot say the valleys between them are
    # empty. neither of them is a shape. day 2 emits a mixture and gets to say
    # both things at once.
    # the valley is the minimum of the density between two preimages, not the
    # midpoint between them. those are different points here because g is not
    # symmetric on the interval, and the midpoint version understates the dip.
    between = (grid > center_roots[0]) & (grid < center_roots[1])
    valley = float(grid[between][np.argmin(center_density[between])])
    at_valley = float(center_density[between].min())
    at_mode = float(np.interp(center_roots[1], grid, center_density))

    print(f"  valley at               : {valley:.4f}")
    print(f"  valley/mode density     : {at_valley / at_mode:.4f}")

    # a real dip, and shallower than the textbook picture of this dataset. at
    # sigma = 0.05 the fold only lifts g about 0.09 above y = 0.5, which is under
    # two sigma, so the modes are marginally resolved rather than cleanly
    # separated. worth pinning now because it is day 2's problem: a mixture has to
    # separate components that overlap this much, and a picture with three
    # obviously distinct spikes in it would have set the wrong expectation.
    assert at_valley < 0.25 * at_mode, (at_valley, at_mode)
    assert at_valley > 0.05 * at_mode, (at_valley, at_mode)

    # the valley is inside the band, which is the precise sense in which the
    # interval points at somewhere the answer provably is not
    assert abs(valley - center_pred) < radius, (valley, center_pred, radius)

    print("day 1 checks passed")
