"""
Day 3: Bayesian A/B testing, visualization, and final conclusions.

We model the per-variant conversion rate with a Beta-Binomial setup. A
Beta(1, 1) prior is essentially uniform on [0, 1] which is a reasonable
weakly-informative default when we don't want to bake in strong beliefs.
The posterior is conjugate: Beta(1 + successes, 1 + failures).

Once we have posterior samples for control and treatment we can answer
the questions decision makers actually care about, without p-values:

  - P(treatment > control)
  - Posterior distribution of the lift (absolute and relative)
  - Credible intervals for the lift
  - Expected loss if we wrongly pick treatment

These are easier to communicate to a PM than "p = 0.043, reject the null".
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
N_SAMPLES = 200_000


def posterior_samples(successes: int, failures: int, n: int = N_SAMPLES) -> np.ndarray:
    """Draw samples from the Beta posterior given observed counts."""
    a = PRIOR_ALPHA + successes
    b = PRIOR_BETA + failures
    return rng.beta(a, b, size=n)


def summarize_posterior(samples: np.ndarray) -> dict:
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def expected_loss(samples_ctrl: np.ndarray, samples_trt: np.ndarray) -> dict:
    """
    Expected loss of each decision. If we choose treatment but control
    is actually better, the loss is (p_ctrl - p_trt) clipped at 0.
    """
    loss_pick_trt = np.maximum(samples_ctrl - samples_trt, 0).mean()
    loss_pick_ctrl = np.maximum(samples_trt - samples_ctrl, 0).mean()
    return {
        "loss_if_pick_treatment": float(loss_pick_trt),
        "loss_if_pick_control": float(loss_pick_ctrl),
    }


def plot_posteriors(samples_ctrl: np.ndarray, samples_trt: np.ndarray, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(samples_ctrl, bins=80, alpha=0.55, label="control", density=True)
    axes[0].hist(samples_trt, bins=80, alpha=0.55, label="treatment", density=True)
    axes[0].set_xlabel("conversion rate")
    axes[0].set_ylabel("posterior density")
    axes[0].set_title("Posterior of conversion rate by variant")
    axes[0].legend()

    lift = samples_trt - samples_ctrl
    axes[1].hist(lift, bins=80, density=True, color="steelblue", alpha=0.7)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    lo, hi = np.quantile(lift, [0.025, 0.975])
    axes[1].axvline(lo, color="red", linestyle=":", linewidth=1)
    axes[1].axvline(hi, color="red", linestyle=":", linewidth=1)
    axes[1].set_xlabel("absolute lift (treatment - control)")
    axes[1].set_ylabel("posterior density")
    axes[1].set_title("Posterior of the lift with 95% credible interval")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(data_path: str = "ab_experiment_data.csv") -> dict:
    df = pd.read_csv(data_path)
    grp = df.groupby("variant")["converted"].agg(["sum", "count"])
    s_ctrl, n_ctrl = int(grp.loc["control", "sum"]), int(grp.loc["control", "count"])
    s_trt, n_trt = int(grp.loc["treatment", "sum"]), int(grp.loc["treatment", "count"])

    samples_ctrl = posterior_samples(s_ctrl, n_ctrl - s_ctrl)
    samples_trt = posterior_samples(s_trt, n_trt - s_trt)

    p_trt_beats_ctrl = float((samples_trt > samples_ctrl).mean())
    lift_abs = samples_trt - samples_ctrl
    lift_rel = (samples_trt - samples_ctrl) / samples_ctrl

    summary = {
        "control": summarize_posterior(samples_ctrl),
        "treatment": summarize_posterior(samples_trt),
        "p_treatment_better": p_trt_beats_ctrl,
        "absolute_lift": summarize_posterior(lift_abs),
        "relative_lift": summarize_posterior(lift_rel),
        "expected_loss": expected_loss(samples_ctrl, samples_trt),
    }

    plot_posteriors(samples_ctrl, samples_trt, "posterior_plots.png")
    return summary


def pretty_print(summary: dict) -> None:
    print("=" * 60)
    print("Bayesian A/B test results")
    print("=" * 60)
    for variant in ("control", "treatment"):
        s = summary[variant]
        print(
            f"  {variant:<10} mean={s['mean']:.4f} "
            f"95% CI=[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
        )
    print()
    print(f"  P(treatment > control) = {summary['p_treatment_better']:.4f}")
    al = summary["absolute_lift"]
    rl = summary["relative_lift"]
    print(
        f"  Absolute lift: mean={al['mean']:.4f} "
        f"95% CI=[{al['ci_low']:.4f}, {al['ci_high']:.4f}]"
    )
    print(
        f"  Relative lift: mean={rl['mean']:.2%} "
        f"95% CI=[{rl['ci_low']:.2%}, {rl['ci_high']:.2%}]"
    )
    el = summary["expected_loss"]
    print(f"  Expected loss if pick treatment: {el['loss_if_pick_treatment']:.5f}")
    print(f"  Expected loss if pick control:   {el['loss_if_pick_control']:.5f}")


if __name__ == "__main__":
    summary = run()
    pretty_print(summary)
