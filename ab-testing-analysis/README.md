# A/B Testing Statistical Analysis

End-to-end walk-through of how to design, analyze, and report on an
online A/B test. The goal is to cover both the standard frequentist
toolkit and a Bayesian alternative on the same synthetic experiment
so the two styles can be compared directly.

## Structure

| File | What it covers |
|------|----------------|
| `01_sample_size_power_analysis.py` | Sample-size calculation, power analysis, and generation of the synthetic experiment data used by the other scripts. |
| `02_frequentist_tests.py` | Two-proportion z-test, chi-square, Welch's t-test, Mann-Whitney U, bootstrap CI for the lift, and an "early peeking" simulation. |
| `03_bayesian_analysis.py` | Beta-Binomial posterior for each variant, posterior of the absolute and relative lift, expected loss, and visualization. |

Run them in order. Day 1 writes `ab_experiment_data.csv`, which days 2
and 3 then read.

## Key takeaways

- **Plan the sample size first.** Detecting a 2pp lift on a 10% baseline
  at 80% power and alpha=0.05 needs roughly 4k users per variant.
  Smaller experiments tend to be underpowered for the effects we
  actually care about.
- **Frequentist and Bayesian usually agree** when the sample is large
  and the prior is weak, but the Bayesian framing answers different
  questions: "what's the probability treatment is better" and
  "what's my expected loss if I ship the wrong variant".
- **Don't peek.** The early-peeking simulation in day 2 shows how the
  false positive rate balloons past the nominal 5% if you re-test
  daily without a sequential-testing correction.
- **Communicate the lift, not just the p-value.** The 95% credible
  interval on the relative lift is what a product manager actually
  uses to make a ship/no-ship call.

## Stack

- NumPy, Pandas, SciPy
- statsmodels (power analysis, proportion tests)
- Matplotlib for posterior plots
