"""
Lift calculation engine.

Computes various lift metrics from causal estimates:
  - Absolute lift (ATE)
  - Relative lift (% change)
  - Lift by segment / subgroup
  - Cumulative lift curves (uplift curves)
  - Qini curves
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiftResult:
    """Container for lift analysis results."""

    absolute_lift: float
    relative_lift_pct: float
    control_rate: float
    treatment_rate: float
    n_treated: int
    n_control: int
    statistical_significance: bool
    p_value: float
    ci_lower: float
    ci_upper: float
    nnt: float  # Number needed to treat


def compute_basic_lift(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> LiftResult:
    """Compute naive (unadjusted) lift between treatment and control."""
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    treatment_rate = treated[outcome_col].mean()
    control_rate = control[outcome_col].mean()

    absolute_lift = treatment_rate - control_rate
    relative_lift = (absolute_lift / control_rate * 100) if control_rate > 0 else 0.0

    # Two-proportion z-test
    n1, n0 = len(treated), len(control)
    p_pool = df[outcome_col].mean()
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0)) if p_pool > 0 else 1e-10
    z = absolute_lift / se
    from scipy import stats
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    nnt = 1.0 / absolute_lift if absolute_lift > 0 else float("inf")

    return LiftResult(
        absolute_lift=round(absolute_lift, 6),
        relative_lift_pct=round(relative_lift, 2),
        control_rate=round(control_rate, 6),
        treatment_rate=round(treatment_rate, 6),
        n_treated=n1,
        n_control=n0,
        statistical_significance=p_value < 0.05,
        p_value=round(p_value, 6),
        ci_lower=round(absolute_lift - 1.96 * se, 6),
        ci_upper=round(absolute_lift + 1.96 * se, 6),
        nnt=round(nnt, 1),
    )


def compute_segment_lift(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    segment_col: str,
) -> pd.DataFrame:
    """Compute lift broken down by a categorical segment variable."""
    results = []
    for seg_val, seg_df in df.groupby(segment_col):
        if seg_df[treatment_col].nunique() < 2:
            continue
        lr = compute_basic_lift(seg_df, treatment_col, outcome_col)
        results.append(
            {
                "segment": seg_val,
                "n_treated": lr.n_treated,
                "n_control": lr.n_control,
                "treatment_rate": lr.treatment_rate,
                "control_rate": lr.control_rate,
                "absolute_lift": lr.absolute_lift,
                "relative_lift_pct": lr.relative_lift_pct,
                "p_value": lr.p_value,
                "significant": lr.statistical_significance,
            }
        )
    return pd.DataFrame(results)


def compute_uplift_curve(
    predicted_cate: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Compute cumulative uplift (gain) curve data.

    Sorts individuals by predicted CATE descending, then computes the
    cumulative uplift as we move from highest to lowest predicted effect.

    Returns a DataFrame suitable for plotting.
    """
    order = np.argsort(-predicted_cate)
    sorted_t = treatment[order]
    sorted_y = outcome[order]

    n = len(order)
    fractions = []
    cum_uplifts = []

    for k in range(1, n + 1, max(1, n // n_bins)):
        top_t = sorted_t[:k]
        top_y = sorted_y[:k]
        t_mask = top_t == 1
        c_mask = top_t == 0

        if t_mask.sum() > 0 and c_mask.sum() > 0:
            uplift = top_y[t_mask].mean() - top_y[c_mask].mean()
        else:
            uplift = 0.0

        fractions.append(k / n)
        cum_uplifts.append(uplift)

    return pd.DataFrame({"fraction_targeted": fractions, "cumulative_uplift": cum_uplifts})


def compute_qini_curve(
    predicted_cate: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Compute Qini curve data (cumulative incremental gains).

    The Qini curve plots the number of incremental conversions as a function
    of the fraction of the population targeted, ordered by predicted uplift.
    """
    order = np.argsort(-predicted_cate)
    sorted_t = treatment[order]
    sorted_y = outcome[order]

    n = len(order)
    fractions = [0.0]
    qini_values = [0.0]

    for k in range(1, n + 1, max(1, n // n_bins)):
        top_t = sorted_t[:k]
        top_y = sorted_y[:k]

        n_t = (top_t == 1).sum()
        n_c = (top_t == 0).sum()

        if n_c > 0 and n_t > 0:
            incremental = top_y[top_t == 1].sum() - top_y[top_t == 0].sum() * (n_t / n_c)
        else:
            incremental = 0.0

        fractions.append(k / n)
        qini_values.append(incremental)

    return pd.DataFrame({"fraction_targeted": fractions, "incremental_gains": qini_values})


def compute_lift_by_decile(
    predicted_cate: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> pd.DataFrame:
    """Compute observed lift within each CATE decile."""
    df = pd.DataFrame(
        {"cate": predicted_cate, "treatment": treatment, "outcome": outcome}
    )
    df["decile"] = pd.qcut(df["cate"], 10, labels=False, duplicates="drop") + 1

    rows = []
    for dec, grp in df.groupby("decile"):
        t = grp[grp["treatment"] == 1]
        c = grp[grp["treatment"] == 0]
        if len(t) > 0 and len(c) > 0:
            obs_lift = t["outcome"].mean() - c["outcome"].mean()
        else:
            obs_lift = 0.0
        rows.append(
            {
                "decile": int(dec),
                "mean_predicted_cate": grp["cate"].mean(),
                "observed_lift": obs_lift,
                "n_treated": len(t),
                "n_control": len(c),
            }
        )
    return pd.DataFrame(rows)
