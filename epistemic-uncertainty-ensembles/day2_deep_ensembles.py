"""
Day 2 of epistemic uncertainty: deep ensembles.

Day 1 ended with two networks from different initializations agreeing on the data
and disagreeing in the gap, and with an explicit note that calling that
disagreement "the epistemic term" was day 2's question and not a thing to assume.
This is that question, and the reason it is a question rather than a formality is
that the disagreement has two possible sources and only one of them is the one
being claimed:

  1. the data does not pin the function down here, so models that fit the data
     equally well differ here - which is epistemic uncertainty, exactly;
  2. two runs of SGD from different random initializations land in different
     places for reasons that have nothing to do with what the data does or does
     not determine - optimization noise, wearing the same clothes.

Both produce a nonzero `Var_theta[mu(x)]`. The decomposition in `day1.decompose`
cannot tell them apart, because it is arithmetic over whatever models it is
handed, and it will report a number either way. So the number itself proves
nothing, and "epistemic uncertainty grows in the gap" is not established by
measuring it in the gap and finding it large. It needs a control.

**The control is an ensemble trained on data with no gap in it.** Same
architecture, same seeds, same optimizer, same everything, and the interval
`(-1.5, 1.5)` filled in. Whatever disagreement that ensemble shows on the same
interval is source (2), because source (1) has been removed by construction. The
gapped ensemble's disagreement is sources (1) and (2) together. The difference is
the signal, and the ratio is how usable it is.

One detail decides whether the control is a control. The gapped dataset draws 600
points from a domain of length 5, and the filled dataset draws from a domain of
length 8. Handing both of them `n = 600` would make the filled ensemble see a
lower density everywhere and confound "has data in the gap" with "has less data
per unit length", in the direction that flatters the result. So the control gets
`600 * 8 / 5 = 960` points and the two ensembles see the same density on the two
outer regions, which are the regions where they are supposed to agree.

Second thing the day settles, which is smaller but was going to bite later: the
ensemble's predictive distribution is a uniform mixture of Gaussians, and this
project spent three days in the MDN project building exactly that object for
exactly the opposite reason. There the mixture was the answer, because the
conditional distribution of the data really had several modes. Here the data is
unimodal Gaussian everywhere by construction - `true_mean` plus `true_sigma`
noise, one mode, no ambiguity - and the mixture's modes are an artefact of
parameter uncertainty. Same functional form, opposite meaning, and it matters
because the mode structure in the gap is not a claim about the data and must not
be read as one.

Third: `Var_theta` over `M` members is an estimator, and with small `M` it is a
bad one. The epistemic number reported at `M = 5` is a sample variance from five
draws. Its own sampling spread is large, and the project's day 4 is going to
compare methods on this number, so the size at which it stops moving needs to be
known now rather than assumed.
"""

import numpy as np
import torch

from day1_the_gap import (
    DOMAIN,
    GAP,
    decompose,
    predict,
    sample_dataset,
    train_gaussian_head,
    true_mean,
    true_sigma,
)


ENSEMBLE_SIZE = 10


def train_ensemble(x, y, size=ENSEMBLE_SIZE, seed_base=100, **kwargs):
    """Train `size` independent `GaussianHead`s on the same data.

    Independent means different initialization and nothing else. No bootstrap
    resampling of the training set, which is the choice Lakshminarayanan et al.
    make and justify empirically, and which matters here for a reason specific to
    this experiment: bootstrapping would give each member a different dataset, so
    members would disagree partly because they saw different data, and that is a
    third source of disagreement on top of the two the control is built to
    separate. Keeping the data fixed means the control isolates exactly the
    optimization component.

    Training is full batch, so there is no data ordering either. The *only* thing
    that differs between members is the seed passed to `torch.manual_seed`, which
    sets the initial weights. That is a cleaner setup than a real deep ensemble
    and it is deliberately the least favourable one for the method - fewer sources
    of diversity means less disagreement, so whatever signal survives here is not
    coming from anything incidental.
    """
    return [
        train_gaussian_head(x, y, seed=seed_base + i, **kwargs)
        for i in range(size)
    ]


def mixture_log_prob(models, x, y):
    """Log density of `y` under the ensemble's uniform Gaussian mixture at `x`.

    `log (1/M) sum_i N(y; mu_i(x), sigma_i(x)^2)`, by log-sum-exp.

    This is the ensemble's actual predictive distribution and it is not the
    Gaussian implied by the mean and total variance that `decompose` returns.
    Scoring against the Gaussian summary instead of the mixture would understate
    the ensemble wherever the members genuinely disagree, which is exactly the
    region under study, so the summary is for reporting and this is for scoring.
    """
    means = np.stack([predict(m, x)[0] for m in models])
    sigmas = np.stack([predict(m, x)[1] for m in models])

    components = (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigmas)
        - 0.5 * ((y[None, :] - means) / sigmas) ** 2
    )

    top = components.max(axis=0)
    return top + np.log(np.exp(components - top).mean(axis=0))


def mixture_density(models, x_point, y_grid):
    """The predictive density at a single `x`, over a grid of `y`, for mode counting."""
    x = np.full_like(y_grid, x_point, dtype=np.float32)
    return np.exp(mixture_log_prob(models, x, y_grid.astype(np.float32)))


def count_modes(density, floor_fraction=0.01):
    """Number of interior local maxima of a 1-D density, ignoring numerical dust.

    `floor_fraction` drops bumps below a fraction of the global peak. Without it
    this counts float noise in the tails as modes, which is not a bug in the model
    and would be a bug in the measurement.
    """
    floor = density.max() * floor_fraction
    interior = density[1:-1]
    rising = interior > density[:-2]
    falling = interior > density[2:]

    return int(np.sum(rising & falling & (interior > floor)))


def region_masks(x):
    """Split a grid into (inside the gap, outside it), as boolean masks."""
    inside = (x > GAP[0]) & (x < GAP[1])
    return inside, ~inside


def summarize(models, x_grid):
    """Mean, aleatoric and epistemic on a grid, plus the two regional averages.

    Returns epistemic and aleatoric as *standard deviations* alongside the
    variances, because every comparison in this project is eventually against a
    noise scale and reading a variance against `true_sigma` is an invitation to
    compare a squared thing with an unsquared one.
    """
    mean, aleatoric, epistemic = decompose(models, x_grid)
    inside, outside = region_masks(x_grid)

    return {
        "mean": mean,
        "aleatoric_var": aleatoric,
        "epistemic_var": epistemic,
        "aleatoric_sd": np.sqrt(aleatoric),
        "epistemic_sd": np.sqrt(epistemic),
        "epistemic_sd_gap": float(np.sqrt(epistemic[inside]).mean()),
        "epistemic_sd_data": float(np.sqrt(epistemic[outside]).mean()),
        "aleatoric_sd_gap": float(np.sqrt(aleatoric[inside]).mean()),
        "aleatoric_sd_data": float(np.sqrt(aleatoric[outside]).mean()),
        "mean_error_gap": float(np.abs(mean - true_mean(x_grid))[inside].max()),
        "mean_error_data": float(np.abs(mean - true_mean(x_grid))[outside].max()),
    }


def epistemic_by_size(models, x_grid, sizes, draws=12, seed=0):
    """How the epistemic estimate moves as the ensemble grows.

    For each `size`, draw `draws` random subsets of the trained pool, compute the
    mean epistemic standard deviation in the gap for each, and report the mean and
    the spread across subsets. The spread is the quantity of interest: it is the
    sampling error of the estimator itself, and it is what says whether a
    difference between two methods on day 4 is a difference between the methods or
    between two draws of five networks.

    Subsets are drawn from one pool rather than trained fresh per size, which
    reuses members across draws and therefore understates the spread somewhat. It
    is the affordable version, and understating the spread is the conservative
    direction for the conclusion being drawn - if it already looks large here, it
    is at least that large.

    Two columns, because the raw trend has a known confound in it. `decompose`
    uses `numpy`'s default `ddof=0`, so `Var_theta` is the biased sample variance
    and its expectation is `(M-1)/M` times the truth. That alone makes the
    estimate rise with `M` whether or not anything real is happening, by a factor
    of `sqrt((M-1)/M)` on the standard deviation - `0.71` at `M = 2` against
    `0.95` at `M = 10`, so about a third. The corrected column multiplies the
    variance by `M/(M-1)` before the square root, and the question is how much of
    the trend survives it. Whatever does survive is real convergence behaviour and
    is the thing day 4 needs to know about.

    `ddof=0` stays in `decompose` itself, deliberately. The law of total variance
    for a uniform mixture of `M` components is exact with the population variance
    over those components - the mixture the ensemble actually defines has that
    variance and not a corrected one. The correction is the right thing when the
    `M` members are being treated as a sample from a posterior, which is what this
    function is asking about, and the wrong thing when the mixture is being scored
    as itself, which is what day 4 will do. Two different questions about the same
    numbers, and conflating them is easier than it looks.
    """
    rng = np.random.default_rng(seed)
    inside, _ = region_masks(x_grid)

    rows = []
    for size in sizes:
        values = []
        for _ in range(draws):
            picked = rng.choice(len(models), size=size, replace=False)
            _, _, epistemic = decompose([models[i] for i in picked], x_grid)
            values.append(float(np.sqrt(epistemic[inside]).mean()))

        correction = np.sqrt(size / (size - 1.0)) if size > 1 else float("nan")
        rows.append(
            (
                size,
                float(np.mean(values)),
                float(np.std(values)),
                float(np.mean(values) * correction),
            )
        )

    return rows


if __name__ == "__main__":
    # single thread is a measured choice and not superstition: these are 64-unit
    # MLPs on 600 points, the per-op work is far below the threading overhead, and
    # the intra-op pool costs a factor of three in wall clock. twenty members at
    # six seconds instead of twenty-one.
    torch.set_num_threads(1)

    grid = np.linspace(DOMAIN[0], DOMAIN[1], 400).astype(np.float32)

    x_train, y_train = sample_dataset(600, seed=0)
    assert not ((x_train > GAP[0]) & (x_train < GAP[1])).any()

    # the control's density is matched, not its count. domain length 8 against the
    # gapped 5, so 600 * 8 / 5.
    filled_n = int(round(600 * (DOMAIN[1] - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0] - (GAP[1] - GAP[0]))))
    x_filled, y_filled = sample_dataset(filled_n, seed=0, gap=(0.0, 0.0))
    assert ((x_filled > GAP[0]) & (x_filled < GAP[1])).any()

    print(f"gapped train: n={len(x_train)}   filled train: n={len(x_filled)}")
    print(f"density check: gapped={len(x_train) / 5.0:.1f}/unit  "
          f"filled={len(x_filled) / 8.0:.1f}/unit")

    print("\ntraining the gapped ensemble ...")
    ensemble = train_ensemble(x_train, y_train, seed_base=100)

    print("training the control ensemble on filled data ...")
    control = train_ensemble(x_filled, y_filled, seed_base=100)

    gapped = summarize(ensemble, grid)
    filled = summarize(control, grid)

    # --- the measurement, and the control ------------------------------------
    print("\n--- epistemic sd, averaged over each region ---")
    print(f"{'':22s}{'in gap':>10s}{'on data':>10s}{'ratio':>10s}")
    for label, row in (("gapped ensemble", gapped), ("control (filled)", filled)):
        ratio = row["epistemic_sd_gap"] / row["epistemic_sd_data"]
        print(f"{label:22s}{row['epistemic_sd_gap']:10.4f}"
              f"{row['epistemic_sd_data']:10.4f}{ratio:10.2f}")

    # the number that decides the day. the control's in-gap disagreement is pure
    # optimization noise, since its training data covers the interval. the gapped
    # ensemble's is that plus whatever the missing data contributes.
    noise_floor = filled["epistemic_sd_gap"]
    signal = gapped["epistemic_sd_gap"]
    print(f"\nin-gap disagreement, gapped / control = {signal / noise_floor:.1f}x")
    print(f"  gapped ensemble : {signal:.4f}")
    print(f"  optimization floor : {noise_floor:.4f}")

    assert signal > 5.0 * noise_floor, (
        "in-gap disagreement is not clearly above the optimization floor - "
        "the epistemic reading would not be attributable to the missing data"
    )

    # and the other half of the control: the two ensembles have to agree where
    # they both have data, or the comparison above is between two different fits
    # rather than between two data regimes.
    print(f"\non-data disagreement, gapped={gapped['epistemic_sd_data']:.4f}  "
          f"control={filled['epistemic_sd_data']:.4f}")
    assert abs(gapped["epistemic_sd_data"] - filled["epistemic_sd_data"]) < 0.05

    # --- the anti-correlation check from day 1 -------------------------------
    # true_sigma is smallest at the centre and largest at the edges, on purpose.
    # a method that is re-reporting the aleatoric term under a different name gets
    # the sign of this backwards, so the check is that epistemic peaks where
    # aleatoric bottoms out.
    print("\n--- shape check: the two terms point opposite ways ---")
    print(f"aleatoric sd  in gap {gapped['aleatoric_sd_gap']:.4f}   "
          f"on data {gapped['aleatoric_sd_data']:.4f}")
    print(f"true sigma    in gap {true_sigma(grid[region_masks(grid)[0]]).mean():.4f}   "
          f"on data {true_sigma(grid[region_masks(grid)[1]]).mean():.4f}")

    assert gapped["aleatoric_sd_gap"] < gapped["aleatoric_sd_data"]
    assert gapped["epistemic_sd_gap"] > gapped["epistemic_sd_data"]

    # --- the mean is still wrong in the gap ----------------------------------
    # this is the point of the whole project stated in two numbers. the ensemble
    # does not fix the mean in the gap - nothing can, the data is not there - it
    # reports that the mean is not to be trusted there.
    print(f"\nmax |mean error| in gap {gapped['mean_error_gap']:.4f}   "
          f"on data {gapped['mean_error_data']:.4f}")
    miss = gapped["mean_error_gap"] / signal
    print(f"in-gap error in units of reported epistemic sd: {miss:.2f}")

    # and this is the day's second result, which is not the one i went looking
    # for. the epistemic term is real, it is attributable to the missing data, and
    # it points the right way - all three established above. it is also *too
    # small*. the worst mean error in the gap sits at two and a half reported
    # standard deviations, on a region where the model has no information at all
    # and the honest posterior is close to "anything the prior allows". the
    # ensemble knows it does not know, and then understates by a factor of two or
    # three how much it does not know.
    #
    # that is the known failure of deep ensembles and it is worth having produced
    # it rather than read it: ten independently initialized networks are not a
    # sample from the posterior over functions, they are a sample from whatever
    # distribution gradient descent from a gaussian init induces, and that
    # distribution is concentrated relative to the posterior. asserted as a
    # measurement of the gap between them rather than as a target to hit, since
    # the number itself is what days 3 and 4 have to improve on.
    assert miss > 1.5, (
        "the reported epistemic sd covers the in-gap error - unexpectedly good, "
        "check the control before believing it"
    )

    # --- the mixture is not the Gaussian summary -----------------------------
    y_grid = np.linspace(-4.0, 4.0, 1200)
    gap_centre = float(np.mean(GAP))
    data_point = 3.0

    modes_gap = count_modes(mixture_density(ensemble, gap_centre, y_grid))
    modes_data = count_modes(mixture_density(ensemble, data_point, y_grid))

    print(f"\nmixture modes at x={gap_centre:+.1f} (gap): {modes_gap}")
    print(f"mixture modes at x={data_point:+.1f} (data): {modes_data}")

    # the MDN project's object, arriving for the opposite reason. the data here is
    # unimodal Gaussian everywhere by construction, so any structure in the gap is
    # about the parameters and not about the conditional distribution. worth an
    # assert rather than a sentence, because a reader arriving from that project
    # will recognise the plot and misread it.
    assert modes_data == 1, "members disagree where the data pins them down"

    # --- scoring: mixture NLL, inside and outside ----------------------------
    # against a single member, which is day 1's model. the ensemble should be
    # roughly a wash where there is data and much better in the gap, and "much
    # better in the gap" is the only claim that needs the ensemble at all.
    x_eval_data, y_eval_data = sample_dataset(400, seed=7)
    x_eval_gap = np.linspace(GAP[0] + 0.1, GAP[1] - 0.1, 400).astype(np.float32)
    rng = np.random.default_rng(7)
    y_eval_gap = (
        true_mean(x_eval_gap) + true_sigma(x_eval_gap) * rng.standard_normal(400)
    ).astype(np.float32)

    print("\n--- mean NLL (lower is better) ---")
    print(f"{'':22s}{'on data':>10s}{'in gap':>10s}")
    for label, models in (("single member", [ensemble[0]]), ("ensemble of 10", ensemble)):
        on_data = -mixture_log_prob(models, x_eval_data, y_eval_data).mean()
        in_gap = -mixture_log_prob(models, x_eval_gap, y_eval_gap).mean()
        print(f"{label:22s}{on_data:10.3f}{in_gap:10.3f}")

    single_gap = -mixture_log_prob([ensemble[0]], x_eval_gap, y_eval_gap).mean()
    ens_gap = -mixture_log_prob(ensemble, x_eval_gap, y_eval_gap).mean()
    assert ens_gap < single_gap, "the ensemble has to pay off somewhere"

    # --- how many members before the estimate stops moving -------------------
    print("\n--- epistemic sd in gap, by ensemble size ---")
    print(f"{'size':>6s}{'mean':>10s}{'spread':>10s}{'rel':>8s}{'bias-corrected':>16s}")
    rows = epistemic_by_size(ensemble, grid, sizes=(2, 3, 5, 8, 10))
    for size, mean_value, spread, corrected in rows:
        rel = spread / mean_value if mean_value else 0.0
        note = "  <- whole pool, one subset" if size == len(ensemble) else ""
        print(f"{size:6d}{mean_value:10.4f}{spread:10.4f}{rel:8.1%}"
              f"{corrected:16.4f}{note}")

    # the size-10 row has a spread of exactly zero because there is one subset of
    # size ten in a pool of ten. it is a point estimate sitting in a column of
    # sampling errors and it should not be read as convergence.
    raw_climb = rows[-1][1] / rows[0][1]
    corrected_climb = rows[-1][3] / rows[0][3]
    print(f"\nclimb from M=2 to M=10: raw {raw_climb:.2f}x, "
          f"bias-corrected {corrected_climb:.2f}x")

    # the conclusion day 4 needs. the ddof=0 bias explains part of the climb and
    # nowhere near all of it, so the estimate is genuinely still moving at ten
    # members and has not converged. day 4 was going to compare three methods on
    # this number; at a relative spread of nearly 30% at M=8 that comparison would
    # be reporting subset-of-the-pool noise as a difference between methods.
    assert corrected_climb > 1.5, (
        "the climb is explained by the ddof=0 bias alone - the estimate has "
        "converged and the size study is not needed"
    )

    print("\nday 2 done")
