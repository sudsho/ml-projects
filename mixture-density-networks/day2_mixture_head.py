"""
Day 2 of mixture density networks.

Day 1 established that the failure is the output object rather than the number:
the point regressor returns a legitimate preimage on the most probable branch,
looks correct under every point metric, and has no way to say the other two
preimages exist. Today replaces the single output with a mixture head, so the
network emits `K` weights, means and scales instead of one number, and replaces
squared error with the negative log-likelihood of the target under that mixture.

The head is three lines of algebra and the loss is one. What took the day is the
loss's numerics, and the thing I came in believing about them is wrong in a way
that matters for day 3.

The standard telling is that you write the mixture NLL through `logsumexp`
because the naive `-log sum_k pi_k N(t | mu_k, sigma_k)` underflows, and that the
stable form keeps the far components alive. The first half is true. The second
half is not, and it is the half I was relying on.

Measured here, with a component sitting 0.48 away from the target at sigma = 0.02:
its log-responsibility is -288, and `exp(-288)` is exactly zero in float32 under
*both* forms. The stable form gives that component a gradient of `-0.0`,
not a small one. `logsumexp` does not resurrect a component that has drifted
away, because the quantity that has underflowed is the softmax responsibility
itself and that is the gradient. A component that wanders far enough from the
data with a small enough scale is dead in float32 and no rearrangement of the
formula brings it back.

What `logsumexp` actually fixes is the case where *every* component is far from
the target, which is a different event: it is an outlier point, not a dead
component. There the naive form sums three exact zeros, takes `log(0)`, and
returns `inf` with `nan` gradients on all three means, so one bad point destroys
the whole batch. The stable form returns 198.1 and a gradient of exactly -1000
on the nearest mean, which is `-(t - mu) / sigma^2` and is the correct pull. So
the stability argument is about outliers and it is worth making for that reason,
but component death survives it and is day 3's problem, needing something other
than a better-conditioned formula.

Two more traps, both measured below rather than asserted:

  - The scale floor is not an epsilon-for-stability. Without it the objective is
    genuinely unbounded below - a component can shrink onto a single point and
    drive the likelihood to infinity, which is the classic non-existence of the
    Gaussian-mixture MLE. What I had not appreciated is the *shape* of the
    descent: NLL falls off as `log sigma`, and `softplus(raw) ~ exp(raw)` for
    very negative raw, so the loss is linear in the unconstrained parameter with
    slope 1. It is a constant-gradient ramp rather than a cliff, so a small
    learning rate does not avoid it, it only slows the walk down it.
  - `log(softmax(z))` and `log_softmax(z)` agree in the forward pass right up to
    the point where one is `-inf`, and a `-inf` log-weight is absorbed by
    `logsumexp` without complaint, so the *loss stays finite and correct*. Only
    the backward pass is `nan`. The symptom is a clean loss curve followed by
    `nan` parameters one step later, which is the worst possible way for a bug
    to present. It needs a logit spread of about 104 in float32, measured below,
    and that is reachable when the network is killing a component.

Today: the mixture head with softmax weights and floored softplus scales, the
log-sum-exp NLL checked against a brute-force integration of its own density,
and the four traps above pinned with numbers. Day 3 trains it, and inherits
component death from here.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from day1_inverse_problem import make_inverse_problem


LOG_2PI = math.log(2.0 * math.pi)

# the narrowest component this dataset can legitimately need. near a preimage the
# conditional bump has scale sigma / |g'|, and g'(x) = 1 + 0.6 pi cos(2 pi x)
# peaks at 1 + 0.6 pi = 2.885, so at sigma = 0.05 nothing below 0.05 / 2.885 is
# real structure. the floor is set an order of magnitude under that.
MIN_SIGMA = 1e-3


class MixtureDensityNetwork(nn.Module):
    """A shared trunk emitting the parameters of a `K`-component Gaussian mixture.

    One linear head per parameter family rather than a single `3K` output sliced
    three ways. The slicing version is equivalent and is what most references do,
    but the three families have completely different natural scales - logits are
    unconstrained and centred, means live in the output range, raw scales sit
    near a fixed initialization - and sharing one weight matrix across them means
    one initialization has to be right for all three. Separate heads cost three
    bias vectors and make the constraint on each one visible at the point it is
    applied.

    The trunk is deliberately the same shape as day 1's point regressor, since
    the argument being carried forward is that the point estimate failed at its
    optimum rather than for want of capacity. Changing the trunk at the same time
    as the output object would give the comparison two moving parts.
    """

    def __init__(self, n_components=3, hidden=64, min_sigma=MIN_SIGMA):
        super().__init__()

        self.n_components = n_components
        self.min_sigma = min_sigma

        self.trunk = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.logit_head = nn.Linear(hidden, n_components)
        self.mean_head = nn.Linear(hidden, n_components)
        self.scale_head = nn.Linear(hidden, n_components)

    def forward(self, x):
        """Return `(logits, mu, sigma)`, each `(batch, K)`.

        Logits rather than weights, and that is not a detail. The loss needs
        `log pi` and never needs `pi`, so normalizing here and taking a log later
        is a round trip through a quantity that can underflow to zero. Returning
        the unnormalized logits lets the loss call `log_softmax` once, which is
        the whole of trap 4 below.

        The scale is `softplus` shifted up by a floor. `exp` is the other usual
        choice and is worse here for a reason that is about this dataset: the
        legitimate scales span 0.017 to about 0.06, all comfortably order 0.01 to
        0.1, and `exp` makes the map from raw parameter to scale multiplicative,
        so the gradient on a raw parameter is proportional to the scale it
        already has. A component that has started to collapse then collapses
        faster. `softplus` is linear in the raw parameter above zero, so it does
        not compound.
        """
        features = self.trunk(x)

        logits = self.logit_head(features)
        mu = self.mean_head(features)
        sigma = F.softplus(self.scale_head(features)) + self.min_sigma

        return logits, mu, sigma


def component_log_prob(mu, sigma, target):
    """`log N(t | mu_k, sigma_k)` for every component, `(batch, K)`.

    Written out rather than built from `torch.distributions.Normal` so the
    normalization term is visible. It has to be inside the `logsumexp` below,
    since `sigma` varies per component and `-log sigma_k` cannot be factored out
    of the sum. Factoring it out is a plausible-looking error that leaves the
    loss decreasing, because it is still a monotone function of the fit for any
    fixed set of scales, and only goes wrong once the components differ in width
    - which on this dataset is exactly the thing being learned.
    """
    z = (target - mu) / sigma

    return -0.5 * (z**2 + LOG_2PI) - torch.log(sigma)


def mixture_nll(logits, mu, sigma, target):
    """Negative log-likelihood of `target` under the mixture, `(batch,)`.

    `-log sum_k pi_k N(t | mu_k, sigma_k)` rearranged as
    `-logsumexp_k (log pi_k + log N_k)`, which is the same number computed
    without ever forming a density. `logsumexp` factors out the largest term
    before exponentiating, so the biggest exponent seen is zero regardless of how
    far the target is from every component.

    `log_softmax` rather than `log(softmax(...))` - see trap 4. The two agree in
    the forward pass, including in the case where the second one is `-inf`, and
    differ only in the gradient.
    """
    log_weights = F.log_softmax(logits, dim=-1)

    return -torch.logsumexp(log_weights + component_log_prob(mu, sigma, target), dim=-1)


def mixture_nll_naive(logits, mu, sigma, target):
    """The direct transcription of the formula, kept because it is the measurement.

    This is not a strawman. It is what the definition says, it is correct in
    exact arithmetic, and it agrees with the stable version to within float32
    rounding across the entire range where both are finite. Keeping it runnable
    is the only way to say where that range ends instead of asserting that it
    ends somewhere.
    """
    weights = F.softmax(logits, dim=-1)

    density = torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )

    return -torch.log((weights * density).sum(dim=-1))


def mixture_pdf(logits, mu, sigma, grid):
    """The mixture density evaluated on a grid of `t`, for one input.

    Only used to check the loss against a brute-force integration of its own
    density. A likelihood that does not integrate to one is not a likelihood, and
    that failure mode is silent - a missing `sqrt(2 pi)` or a dropped `1/sigma`
    shifts the loss by a constant and leaves the training curve looking normal.
    """
    weights = F.softmax(logits, dim=-1).reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    grid = grid.reshape(1, -1)

    bumps = torch.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )

    return (weights * bumps).sum(dim=0)


def first_failing_spread(lo=1.0, hi=400.0, tol=1e-3):
    """Smallest logit spread at which `log(softmax(z))` produces a `nan` gradient.

    Bisected rather than quoted, because the answer is a property of float32's
    subnormal range and not of anything in this project, and a number I remember
    is a number I can be wrong about. Monotone in the spread, so bisection is
    valid: a larger gap only pushes the small entry further under.
    """

    def is_nan_grad(spread):
        z = torch.tensor([[0.0, -spread]], requires_grad=True)
        picked = (torch.log(F.softmax(z, dim=-1)) * torch.tensor([[1.0, 0.0]])).sum()
        grad = torch.autograd.grad(picked, z)[0]
        return bool(torch.isnan(grad).any())

    assert not is_nan_grad(lo), lo
    assert is_nan_grad(hi), hi

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_nan_grad(mid):
            hi = mid
        else:
            lo = mid

    return hi


if __name__ == "__main__":
    torch.manual_seed(0)

    NOISE = 0.05
    inputs, targets, _ = make_inverse_problem(3000, noise_scale=NOISE, seed=0)

    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)

    model = MixtureDensityNetwork(n_components=3)
    logits, mu, sigma = model(x)

    print(f"samples                   : {len(inputs)}")
    print(f"parameter shapes          : {tuple(logits.shape)} x3")

    # the constraints, at initialization, before any of them have been trained on
    weights = F.softmax(logits, dim=-1)
    assert torch.allclose(weights.sum(-1), torch.ones(len(x)), atol=1e-5)
    assert (weights > 0).all()
    assert (sigma > model.min_sigma).all()

    print(f"weight row sums           : {weights.sum(-1).min():.6f} to "
          f"{weights.sum(-1).max():.6f}")
    print(f"sigma range at init       : {sigma.min():.4f} to {sigma.max():.4f}")

    # is the loss a likelihood at all. the density it implies has to integrate to
    # one, and a dropped constant would be invisible in the loss curve.
    # wide enough that the initialized components, whose scales are order one,
    # are not clipped. the first version of this ran -2 to 3, lost 0.7% of the
    # mass off the ends, and read as a broken normalizer rather than a short grid.
    grid = torch.linspace(-12.0, 12.0, 60001)
    for row in [0, 700, 1500, 2999]:
        pdf = mixture_pdf(logits[row], mu[row], sigma[row], grid)
        area = torch.trapezoid(pdf, grid).item()
        assert abs(area - 1.0) < 1e-4, (row, area)
    print(f"mixture pdf integrates to : {area:.6f}")

    # and the loss agrees with a brute-force -log of that same density, which
    # catches a sign error or a misplaced normalizer that normalization alone
    # would not
    row = 700
    pdf = mixture_pdf(logits[row], mu[row], sigma[row], grid)
    at_target = float(np.interp(t[row].item(), grid.numpy(), pdf.detach().numpy()))
    from_loss = mixture_nll(logits[row], mu[row], sigma[row], t[row]).item()
    assert abs(from_loss - (-math.log(at_target))) < 1e-3, (from_loss, at_target)
    print(f"nll vs -log pdf(t)        : {from_loss:.5f} vs {-math.log(at_target):.5f}")

    # ---- trap 1: the naive form is fine until it is not, and the boundary is
    # not initialization. at init the components have not separated and the
    # scales are order one, so every exponent is small and the two forms agree to
    # rounding. anyone checking the naive loss on step zero sees it pass.
    naive = mixture_nll_naive(logits, mu, sigma, t)
    stable = mixture_nll(logits, mu, sigma, t)

    print("\ntrap 1: naive vs logsumexp at initialization")
    print(f"  max |naive - stable|    : {(naive - stable).abs().max():.3e}")
    print(f"  both finite             : {bool(torch.isfinite(naive).all())}")
    assert torch.allclose(naive, stable, atol=1e-5)

    # the boundary, found by shrinking sigma with the target held 0.5 from the
    # nearest mean. this is not a contrived configuration: on this dataset the
    # outer preimages sit ~0.5 apart and their branch widths are sigma / |g'|
    # with |g'| up to 2.885, so scales of 0.017 are what a correct fit wants.
    far_logits = torch.zeros(1, 3)
    far_mu = torch.tensor([[0.02, 0.10, 0.98]])
    far_t = torch.tensor([[0.50]])

    print("\n  as the scales shrink, target 0.5 from the nearest mean")
    breakdown = None
    for s in [0.20, 0.10, 0.05, 0.04, 0.035, 0.03, 0.02]:
        scales = torch.full((1, 3), s)
        n = mixture_nll_naive(far_logits, far_mu, scales, far_t).item()
        v = mixture_nll(far_logits, far_mu, scales, far_t).item()
        print(f"    sigma={s:5.3f}  naive={n:9.3f}  stable={v:9.3f}")
        if breakdown is None and not math.isfinite(n):
            breakdown = s

    assert breakdown is not None, "naive form never broke down"
    assert 0.02 <= breakdown <= 0.05, breakdown
    print(f"  naive returns inf from  : sigma = {breakdown:.3f}")

    # so the naive loss survives initialization and starts returning inf partway
    # through training, once the scales have tightened. it presents as a training
    # instability rather than as a wrong formula, which is why it is worth the
    # measurement.

    # ---- trap 2: what logsumexp does and does not buy. the correction.
    #
    # all three components far from the target: naive gives inf and nan gradients
    # on every mean, so a single outlier point takes out the batch. stable gives
    # a large finite loss and the right gradient on the nearest component.
    scales = torch.full((1, 3), 0.02)

    mu_a = far_mu.clone().requires_grad_(True)
    loss_stable = mixture_nll(far_logits, mu_a, scales, far_t)
    grad_stable = torch.autograd.grad(loss_stable.sum(), mu_a)[0]

    mu_b = far_mu.clone().requires_grad_(True)
    loss_naive = mixture_nll_naive(far_logits, mu_b, scales, far_t)
    grad_naive = torch.autograd.grad(loss_naive.sum(), mu_b)[0]

    print("\ntrap 2: every component far from the target")
    print(f"  naive  loss / grad      : {loss_naive.item():.3f} / "
          f"{[round(v, 1) for v in grad_naive.ravel().tolist()]}")
    print(f"  stable loss / grad      : {loss_stable.item():.3f} / "
          f"{[round(v, 1) for v in grad_stable.ravel().tolist()]}")

    assert not math.isfinite(loss_naive.item())
    assert torch.isnan(grad_naive).all()
    assert math.isfinite(loss_stable.item())
    assert torch.isfinite(grad_stable).all()

    # the surviving gradient is the exact newton pull toward the nearest mean,
    # -(t - mu) / sigma^2, which is what says the stable form is doing something
    # meaningful here rather than merely not crashing
    nearest = int(torch.argmin((far_mu - far_t).abs()))
    expected = -((far_t - far_mu)[0, nearest] / scales[0, nearest] ** 2).item()
    assert abs(grad_stable[0, nearest].item() - expected) < 1e-3, (
        grad_stable[0, nearest].item(), expected)
    print(f"  pull on nearest mean    : {grad_stable[0, nearest]:.1f} "
          f"(= -(t - mu)/sigma^2 = {expected:.1f})")

    # and now the part i had wrong. one component near, two far. the far ones do
    # NOT get rescued by the stable form - their responsibilities are exp(-288),
    # which is zero in float32 either way, so they receive exactly zero gradient
    # under both. logsumexp fixes outliers, not component death.
    near_mu = torch.tensor([[0.505, 0.02, 0.98]], requires_grad=True)
    near_loss = mixture_nll(far_logits, near_mu, scales, far_t)
    near_grad = torch.autograd.grad(near_loss.sum(), near_mu)[0]

    print("\n  one component near, two far (the case i expected to be rescued)")
    print(f"  stable grad on means    : {near_grad.ravel().tolist()}")

    assert near_grad[0, 0].item() != 0.0
    assert near_grad[0, 1].item() == 0.0 and near_grad[0, 2].item() == 0.0

    log_resp = F.log_softmax(
        F.log_softmax(far_logits, -1) + component_log_prob(near_mu, scales, far_t), -1)
    print(f"  log-responsibilities    : "
          f"{[round(v, 1) for v in log_resp.detach().ravel().tolist()]}")

    # exp of that is zero in float32, and exp of that IS the gradient weight, so
    # the two far components are dead. day 3 inherits this.
    assert log_resp[0, 1].item() < -250, log_resp

    # ---- trap 3: the scale floor is not an epsilon. without it the objective is
    # unbounded below and the descent is a constant-slope ramp, not a cliff.
    print("\ntrap 3: unbounded likelihood without a floor on sigma")
    unfloored = []
    for raw in [0.0, -5.0, -10.0, -20.0, -40.0]:
        s = F.softplus(torch.tensor(raw))
        loss = mixture_nll(
            torch.zeros(1, 3),
            torch.tensor([[0.5, 0.2, 0.8]]),
            torch.tensor([[s, 0.2, 0.2]]),
            torch.tensor([[0.5]]),
        ).item()
        unfloored.append((raw, s.item(), loss))
        print(f"  raw={raw:6.1f}  sigma={s.item():.3e}  nll={loss:8.4f}")

    # the tell: past the softplus knee the loss falls by 1.0 per unit of raw
    # parameter. gradient descent walks down it at constant speed, so a smaller
    # learning rate buys time and not safety.
    slope = (unfloored[-1][2] - unfloored[-2][2]) / (unfloored[-1][0] - unfloored[-2][0])
    print(f"  d(nll)/d(raw) far down  : {slope:.4f}")
    assert abs(slope - 1.0) < 0.01, slope

    # with the floor in place the same collapse is bounded, and the bound is set
    # by the floor rather than by the optimizer giving up
    floored = mixture_nll(
        torch.zeros(1, 3),
        torch.tensor([[0.5, 0.2, 0.8]]),
        torch.tensor([[MIN_SIGMA, 0.2, 0.2]]),
        torch.tensor([[0.5]]),
    ).item()
    print(f"  floored at {MIN_SIGMA:g}       : {floored:.4f}")
    assert math.isfinite(floored)
    assert floored > unfloored[-1][2]

    # and the floor has to be checked against the data rather than picked by
    # habit: the narrowest branch this dataset can want is sigma / max|g'|
    narrowest = NOISE / (1.0 + 0.6 * math.pi)
    print(f"  narrowest real component: {narrowest:.4f} = {narrowest / MIN_SIGMA:.0f}x "
          f"the floor")
    assert MIN_SIGMA < narrowest / 10.0, (MIN_SIGMA, narrowest)

    # ---- trap 4: log(softmax) has a correct forward pass and a nan backward one
    spread = first_failing_spread()

    print("\ntrap 4: log(softmax) vs log_softmax")
    print(f"  logit spread that kills : {spread:.2f}")

    z = torch.tensor([[0.0, -spread - 1.0]], requires_grad=True)
    bad_log_weights = torch.log(F.softmax(z, dim=-1))
    good_log_weights = F.log_softmax(z, dim=-1)

    pair_mu = torch.tensor([[0.5, 0.9]])
    pair_sigma = torch.tensor([[0.05, 0.05]])
    pair_t = torch.tensor([[0.5]])
    comp = component_log_prob(pair_mu, pair_sigma, pair_t)

    bad_loss = -torch.logsumexp(bad_log_weights + comp, dim=-1)
    good_loss = -torch.logsumexp(good_log_weights + comp, dim=-1)

    print(f"  loss via log(softmax)   : {bad_loss.item():.6f}")
    print(f"  loss via log_softmax    : {good_loss.item():.6f}")

    # the forward pass is not merely close, it is identical. logsumexp absorbs a
    # -inf log-weight by dropping that component, which is the right answer.
    assert bad_loss.item() == good_loss.item()
    assert math.isfinite(bad_loss.item())

    bad_grad = torch.autograd.grad(bad_loss.sum(), z, retain_graph=True)[0]
    z2 = torch.tensor([[0.0, -spread - 1.0]], requires_grad=True)
    good_grad = torch.autograd.grad(
        (-torch.logsumexp(F.log_softmax(z2, -1) + comp, -1)).sum(), z2)[0]

    print(f"  grad via log(softmax)   : {bad_grad.ravel().tolist()}")
    print(f"  grad via log_softmax    : {good_grad.ravel().tolist()}")

    assert torch.isnan(bad_grad).any()
    assert torch.isfinite(good_grad).all()

    # so the failure is invisible in the loss curve and shows up as nan
    # parameters on the following step, one layer removed from its cause

    # ---- end to end: the head and the loss on the real data, a few steps, and
    # everything finite. not training - that is day 3 - just evidence that the
    # four traps above are avoided by the code as written rather than in theory.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nend to end on the day 1 dataset")
    for step in range(200):
        optimizer.zero_grad()
        loss = mixture_nll(*model(x), t).mean()
        loss.backward()

        grads = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        assert torch.isfinite(grads).all(), step

        optimizer.step()

        if step % 50 == 0 or step == 199:
            with torch.no_grad():
                _, _, s_now = model(x)
            print(f"  step {step:3d}  nll {loss.item():7.4f}  "
                  f"sigma {s_now.min():.4f} to {s_now.max():.4f}")

    with torch.no_grad():
        final_logits, final_mu, final_sigma = model(x)
        final = mixture_nll(final_logits, final_mu, final_sigma, t).mean().item()

    # a mean NLL below zero is the density exceeding one somewhere, which is the
    # first sign the mixture is concentrating rather than hedging. it is not yet
    # evidence the components have found separate branches, and today does not
    # claim that.
    assert math.isfinite(final)
    assert final < 0.0, final
    assert (final_sigma > MIN_SIGMA).all()

    # the scales have already dropped into the range the geometry says is real,
    # which is also the range where the naive loss would have started returning
    # inf. two hundred steps.
    print(f"  scales after 200 steps  : {final_sigma.min():.4f} to "
          f"{final_sigma.max():.4f}, breakdown was {breakdown:.3f}")
    assert final_sigma.min() < breakdown, (final_sigma.min().item(), breakdown)

    print("day 2 checks passed")
