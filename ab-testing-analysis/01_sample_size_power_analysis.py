"""
Day 1: Sample size calculation, power analysis, and experiment design.

Goal: figure out how many users we need per variant to reliably detect a
realistic lift (say 2pp on a 10% baseline conversion) at alpha=0.05 with
80% power. Also generate a synthetic experiment dataset we'll re-use across
the next two days for hypothesis testing and Bayesian analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def required_sample_size_proportions(p1, p2, alpha=0.05, power=0.8, ratio=1.0):
    """Sample size per group for comparing two proportions (two-sided)."""
    effect_size = proportion_effectsize(p2, p1)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def required_sample_size_means(mean_diff, std, alpha=0.05, power=0.8):
    """Sample size per group for comparing two means with pooled std."""
    effect_size = mean_diff / std
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def power_curve(p1, p2_range, n, alpha=0.05):
    """Power as a function of treatment conversion rate, for fixed n."""
    analysis = NormalIndPower()
    powers = []
    for p2 in p2_range:
        es = abs(proportion_effectsize(p2, p1))
        powers.append(analysis.solve_power(effect_size=es, nobs1=n, alpha=alpha))
    return np.array(powers)


def simulate_experiment(n_per_group, p_control, p_treatment, seed=RANDOM_STATE):
    """Generate a synthetic A/B experiment with conversion + revenue per user.

    Returns a shuffled DataFrame with columns:
      user_id   - unique integer id
      variant   - "control" or "treatment"
      converted - 0/1 outcome from a Bernoulli draw
      revenue   - 0 if not converted, else gamma-distributed AOV
    """
    rng = np.random.default_rng(seed)
    control = rng.binomial(1, p_control, n_per_group)
    treatment = rng.binomial(1, p_treatment, n_per_group)
    # revenue conditional on conversion: gamma-distributed average order value
    rev_control = control * rng.gamma(shape=2.0, scale=25.0, size=n_per_group)
    rev_treatment = treatment * rng.gamma(shape=2.0, scale=27.0, size=n_per_group)

    df = pd.DataFrame(
        {
            "user_id": np.arange(2 * n_per_group),
            "variant": np.r_[
                np.full(n_per_group, "control"),
                np.full(n_per_group, "treatment"),
            ],
            "converted": np.r_[control, treatment].astype(int),
            "revenue": np.r_[rev_control, rev_treatment],
        }
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def design_summary(p_control, mde, alpha, power):
    p_treatment = p_control + mde
    n = required_sample_size_proportions(p_control, p_treatment, alpha, power)
    return {
        "baseline_conversion": p_control,
        "min_detectable_effect": mde,
        "alpha": alpha,
        "power": power,
        "n_per_group": n,
        "total_n": 2 * n,
    }


def main():
    p_control = 0.10
    mde = 0.02  # absolute lift we care about

    design = design_summary(p_control, mde, alpha=0.05, power=0.8)
    print("Experiment design")
    print("-" * 40)
    for k, v in design.items():
        print(f"  {k}: {v}")

    # power curve for the chosen n
    n = design["n_per_group"]
    p2s = np.linspace(p_control, p_control + 0.05, 25)
    powers = power_curve(p_control, p2s, n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p2s, powers, marker="o", linewidth=2)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.6, label="80% power")
    ax.axvline(p_control + mde, color="gray", linestyle=":", alpha=0.6,
               label=f"target p2={p_control+mde:.2f}")
    ax.set_xlabel("treatment conversion rate")
    ax.set_ylabel("statistical power")
    ax.set_title(f"power curve (n={n} per group, baseline={p_control})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("power_curve.png", dpi=120)
    plt.close(fig)

    # generate synthetic data for downstream analysis
    df = simulate_experiment(n, p_control=0.10, p_treatment=0.118)
    df.to_csv("ab_experiment_data.csv", index=False)
    print(f"\nsaved {len(df)} rows to ab_experiment_data.csv")

    # quick sanity check on observed rates
    rates = df.groupby("variant")["converted"].mean()
    print("\nobserved conversion rates:")
    print(rates.round(4))


if __name__ == "__main__":
    main()
