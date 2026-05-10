"""
Day 2: Frequentist hypothesis testing on the synthetic A/B experiment.

Loads ab_experiment_data.csv (produced by day 1) and runs the standard
battery: two-proportion z-test and chi-square for the conversion metric,
Welch's t-test and Mann-Whitney U for revenue, plus a non-parametric
bootstrap CI for the lift in conversion. Also produces a small "early
peeking" simulation to remind ourselves why repeated testing inflates
the false positive rate.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def two_proportion_test(df: pd.DataFrame) -> dict:
    """Two-proportion z-test on the converted indicator."""
    grp = df.groupby("variant")["converted"].agg(["sum", "count"])
    successes = grp["sum"].loc[["control", "treatment"]].to_numpy()
    nobs = grp["count"].loc[["control", "treatment"]].to_numpy()
    z, p = proportions_ztest(successes, nobs, alternative="two-sided")

    p_ctrl = successes[0] / nobs[0]
    p_trt = successes[1] / nobs[1]
    lift = p_trt - p_ctrl

    ci_ctrl = proportion_confint(successes[0], nobs[0], alpha=0.05, method="wilson")
    ci_trt = proportion_confint(successes[1], nobs[1], alpha=0.05, method="wilson")

    return {
        "p_control": p_ctrl,
        "p_treatment": p_trt,
        "absolute_lift": lift,
        "relative_lift": lift / p_ctrl if p_ctrl > 0 else np.nan,
        "z_stat": z,
        "p_value": p,
        "ci_control_95": ci_ctrl,
        "ci_treatment_95": ci_trt,
    }


def chi_square_test(df: pd.DataFrame) -> dict:
    """Chi-square test of independence between variant and conversion."""
    table = pd.crosstab(df["variant"], df["converted"])
    chi2, p, dof, expected = stats.chi2_contingency(table.values, correction=False)
    return {"chi2": chi2, "p_value": p, "dof": dof, "table": table}


def revenue_tests(df: pd.DataFrame) -> dict:
    """Welch's t-test and Mann-Whitney U on per-user revenue."""
    rev_ctrl = df.loc[df["variant"] == "control", "revenue"].to_numpy()
    rev_trt = df.loc[df["variant"] == "treatment", "revenue"].to_numpy()

    t_stat, t_p = stats.ttest_ind(rev_trt, rev_ctrl, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(rev_trt, rev_ctrl, alternative="two-sided")

    return {
        "mean_control": rev_ctrl.mean(),
        "mean_treatment": rev_trt.mean(),
        "welch_t": t_stat,
        "welch_p": t_p,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": u_p,
    }


def bootstrap_lift_ci(df: pd.DataFrame, n_boot: int = 5000, seed: int = RANDOM_STATE) -> dict:
    """Percentile bootstrap CI for the absolute lift in conversion."""
    rng = np.random.default_rng(seed)
    ctrl = df.loc[df["variant"] == "control", "converted"].to_numpy()
    trt = df.loc[df["variant"] == "treatment", "converted"].to_numpy()
    n_c, n_t = len(ctrl), len(trt)

    lifts = np.empty(n_boot)
    for i in range(n_boot):
        c_sample = rng.choice(ctrl, size=n_c, replace=True)
        t_sample = rng.choice(trt, size=n_t, replace=True)
        lifts[i] = t_sample.mean() - c_sample.mean()

    lo, hi = np.percentile(lifts, [2.5, 97.5])
    return {
        "mean_lift": lifts.mean(),
        "ci_lower": lo,
        "ci_upper": hi,
        "n_boot": n_boot,
    }


def early_peeking_simulation(n_per_group=10_000, peeks=20, n_runs=2000, seed=RANDOM_STATE):
    """
    Simulate an A/A test (same conversion rate in both groups) with repeated
    significance checks. Reports the empirical false positive rate when a tester
    stops the experiment as soon as p < 0.05 at ANY peek.
    """
    rng = np.random.default_rng(seed)
    p = 0.10
    checkpoints = np.linspace(n_per_group / peeks, n_per_group, peeks).astype(int)

    false_positives = 0
    for _ in range(n_runs):
        ctrl = rng.binomial(1, p, n_per_group)
        trt = rng.binomial(1, p, n_per_group)
        for n in checkpoints:
            s = np.array([ctrl[:n].sum(), trt[:n].sum()])
            obs = np.array([n, n])
            _, pv = proportions_ztest(s, obs, alternative="two-sided")
            if pv < 0.05:
                false_positives += 1
                break

    return {
        "n_runs": n_runs,
        "peeks": peeks,
        "fpr_with_peeking": false_positives / n_runs,
        "expected_alpha": 0.05,
    }


def main():
    df = pd.read_csv("ab_experiment_data.csv")
    print(f"loaded {len(df)} rows, variants: {df['variant'].unique().tolist()}")

    print("\n[ conversion - two-proportion z-test ]")
    prop_res = two_proportion_test(df)
    for k, v in prop_res.items():
        print(f"  {k}: {v}")

    print("\n[ conversion - chi-square ]")
    chi_res = chi_square_test(df)
    print(f"  chi2={chi_res['chi2']:.4f}  p={chi_res['p_value']:.4g}  dof={chi_res['dof']}")

    print("\n[ revenue ]")
    rev_res = revenue_tests(df)
    for k, v in rev_res.items():
        print(f"  {k}: {v}")

    print("\n[ bootstrap CI for lift in conversion ]")
    boot_res = bootstrap_lift_ci(df, n_boot=5000)
    print(f"  mean lift = {boot_res['mean_lift']:.4f}")
    print(f"  95% CI = [{boot_res['ci_lower']:.4f}, {boot_res['ci_upper']:.4f}]  "
          f"(n_boot={boot_res['n_boot']})")

    # diagnostic plot - bootstrap distribution of the lift
    rng = np.random.default_rng(RANDOM_STATE)
    ctrl = df.loc[df["variant"] == "control", "converted"].to_numpy()
    trt = df.loc[df["variant"] == "treatment", "converted"].to_numpy()
    lifts = np.array([
        rng.choice(trt, size=len(trt), replace=True).mean()
        - rng.choice(ctrl, size=len(ctrl), replace=True).mean()
        for _ in range(5000)
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lifts, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.6, label="no effect")
    ax.axvline(prop_res["absolute_lift"], color="red", linewidth=2, label="observed lift")
    ax.set_xlabel("bootstrap lift (treatment - control)")
    ax.set_ylabel("count")
    ax.set_title("bootstrap distribution of the conversion lift")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("bootstrap_lift.png", dpi=120)
    plt.close(fig)

    print("\n[ early-peeking simulation (A/A test) ]")
    peek_res = early_peeking_simulation(n_runs=500)  # smaller for speed
    for k, v in peek_res.items():
        print(f"  {k}: {v}")
    print("  -> this is why fixed-horizon experiments matter")


if __name__ == "__main__":
    main()
