# Annealed Importance Sampling: Estimating log Z from Outside q

Where the VI project breaks. Every gap it measured was `log Z - ELBO` against a
closed-form normaliser, and the one estimate of `log Z` it built came from q
itself: 0.33 nats low on the funnel from the reverse-KL fit, 0.46 after PSIS.
Annealed importance sampling built from scratch in NumPy on the same
closed-form targets - a geometric path of densities from q to p, MCMC moves
along it, and weights whose mean is unbiased for Z while their log is biased
low. Thermodynamic integration as the deterministic reading of the same path,
bidirectional Monte Carlo as a sandwich on `log Z`, and then the funnel again,
started from VI day 4's q.

## Idea

The VI project's sentence was that the ELBO is exact about q and says nothing
about where q is not. AIS is the standard answer: do not stop at q, walk from it
to p and pay for the walk with MCMC. The question this project was picked to
answer is whether a path out of q escapes the blind spot or only moves it onto
the path.

That is only checkable where `log Z` is known, so every target here has it in
closed form. On the Gaussian targets of days 1 to 3 every density on the path is
Gaussian as well, which makes the exact-transition bias, the log-weight
variance and the second moment of the weight closed forms, and the sampler is
scored against them before anything is read off it. On day 4 there are no closed
forms along the path, only at its ends.

**With exact transitions the bias of log Z is a quadrature error and the path
decides it through one number. With real transitions the bias is set by where
the chains fail to go, which is the blind spot again, one level down.**

## Layout

- `day1_ais_gaussians.py` - the geometric path in closed form, the bias two
  ways (moments and a sum of KLs), the ELBO at T = 1, the second moment of the
  weight as a product of normalisers past `beta = 1`, and MALA transitions.
- `day2_thermodynamic_integration.py` - AIS as TI's left Riemann sum, the
  trapezoid rule, the moment-averaged path as the geometric one mirrored, and
  the `J / L^2` schedule gain.
- `day3_bidirectional_mc.py` - reverse AIS from exact draws of p, the sandwich
  and its width, and the same thing started from states that are only close
  to p.
- `day4_funnel_ais.py` - Neal's funnel from VI day 4's reverse-KL q and from the
  moment-matched q, HMC transitions with MCMC day 4's divergence count, the
  step size, and the sandwich on a target where the chains miss mass.

## What the days actually show

**Day 1: at one temperature AIS is the ELBO, and the head start does not
survive.** `E[log w] = ELBO` exactly at T = 1, and with exact transitions the
bias equals the sum of the KLs between neighbouring temperatures to 5.5e-14. It
falls like `J / 2T` with J the integral of `Var_beta(g)`. The CAVI start is 10x
closer than N(0, I) in KL and ends 1.13x less biased at T = 1000, because
`J = KL(q || p) + KL(p || q)` and the CAVI q is 3.6x further the other way. Where
`E[w^2]` is infinite, which from the CAVI start is every T up to 52, the
unbiased estimate of Z reads 0.77 nats low at T = 1. MALA at 94% acceptance is
8.2x the exact-transition bias at T = 100 and 35x at T = 1000. Three of five
predictions wrong.

**Day 2: the bias is a quadrature error.** AIS with exact transitions is TI's
left Riemann sum, asserted at every T, and the trapezoid rule takes the bias
from 0.138 to 0.011 at T = 100 and leaves the noise where it was. J is the same
on the moment-averaged path, which is the geometric one mirrored in beta, so the
day 1 explanation that the geometric path holds q narrow until the last tenth
was wrong and the docstring says so. A schedule buys at most `J / L^2`, 2.454
from the CAVI start. Under MALA the two paths that exact transitions cannot tell
apart are 2x apart.

**Day 3: the sandwich is a sandwich only from exact draws of p.** Reverse AIS
from p sits above `log Z` by the KLs the other way, so on a uniform geometric
schedule the width is `J / T` exactly. From the CAVI start the upper side is
10.4x the lower at T = 1, and the width reports the ELBO's gap 11.4x too large.
Under MALA it never under-reports. Started from the forward run's final states
instead of exact draws, both sides fall below `log Z` at T = 100 and the
interval excludes the truth.

**Day 4: on the funnel the path gets to about 0.09 nats and stops.** From the
reverse-KL q with HMC the forward bias is 0.446, 0.241 and 0.135 at T = 10, 100
and 1000, 1.8x per decade where a sampler that kept up would give 10x. p has
8.3% of its mass below v = -4.16 and the final states have 4.75% at best. The log
of the mean weight sits at -0.08 to -0.11 at every T, so against plain IS's
-0.33 the path buys a factor of about 3.5 and then nothing: a mean of weights
cannot put back mass no run reached. No step size gets the bias under 0.13.
The sandwich from exact draws of p is 18x to 32x wider than the actual error,
and from the forward finals it lands entirely below `log Z` again, as on day 3.

The moment-matched q, which VI day 4 found 7x better as an importance
proposal, is thousands of nats off on `E log w` and the closest of anything on
`log mean w` at T = 100. The ELBO and the AIS lower bound prefer the reverse-KL
q by thousands of nats, and the log of the mean weight prefers the other one.

## The funnel, beside the two projects before it

`sigma_v = 3`, `d = 3`, `log Z = 0`. Errors are `estimate - log Z`.

| method | from | error | what reports the failure |
|---|---|---|---|
| ELBO (VI day 4) | reverse-KL q | -1.151 | nothing in the trace |
| plain IS, 4000 draws (VI day 4) | reverse-KL q | -0.330 | k-hat 0.880, cannot rank |
| PSIS (VI day 4) | reverse-KL q | -0.463 | the same k-hat |
| AIS `E log w`, T = 1000, HMC | reverse-KL q | -0.135 | 892 divergences, 709 past beta 0.9 |
| AIS `log mean w`, T = 1000, HMC | reverse-KL q | -0.086 | nothing separate |
| BDMC upper side from exact p, T = 1000 | reverse-KL q | +2.224 | 38100 reverse divergences |
| BDMC upper side from forward finals, T = 1000 | reverse-KL q | -0.045 | nothing: both sides below |
| AIS `log mean w`, T = 100, HMC | moment-matched q | -0.008 | `E log w` off by 4209 |

The divergence count is still the one number that is large where the chains
fail, and it still carries no scale.

## Key design choices

**Normalised targets only.** `log Z` is known on every day, so every estimate
is scored against it rather than against another estimate.

**Closed forms before simulation.** On days 1 to 3 the simulated sides sit
inside 2 to 4 standard errors of the exact-transition formulas before any MALA
number is read. Day 4 checks the T = 1 gap against VI's closed form, 1.1518
against 1.151293.

**One sampler per question.** MALA on the Gaussians, where the question is the
path; HMC on the funnel, with the MCMC project's divergence definition, so the
counts mean the same thing in both projects.

**Predictions written into the files before running them.** Across the four
days, ten right, eight wrong, three half, and the docstrings say which.

**Fixed seeds throughout.**

## Where it breaks

**No resampling.** Every run carries its own weight from q to p, and a run that
never reaches the neck contributes nothing there however many temperatures it
walks through. Sequential Monte Carlo samplers resample between temperatures and
pick the next temperature from the effective sample size, which is the particle
filter project's machinery pointed at a path instead of at time. Whether that
helps on the funnel, where the missing mass is in runs that never got there
rather than runs with bad weights, is not measured here.

**The step size is fixed along the path.** One `eps` for every beta, on a target
whose local scale spans a factor of 400. The step-size sweep shows the trade and
does not escape it.

**The sandwich needs exact draws of p.** On both the Gaussians and the funnel,
replacing them with anything a real model would have gave an interval below the
truth.

**Small and closed-form.** `d <= 8` Gaussians and a three-dimensional funnel.
No posterior from data.

## Running

```bash
python day1_ais_gaussians.py
python day2_thermodynamic_integration.py
python day3_bidirectional_mc.py
python day4_funnel_ais.py
```

NumPy only. Day 3 imports days 1 and 2, and day 4 imports the funnel from the VI
project's day 4. About 70s, 45s, 35s and 40s on this machine.
