"""
Lift and uplift curve calculators for Rep-HCP outreach causal inference.

Provides naive lift, segment-level lift, uplift curves, Qini curves,
and decile-level analysis.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LiftResult:
    """Container for a lift estimate."""
    absolute_lift: float
    relative_lift_pct: float
    control_mean: float
    treatment_mean: float
    n_treated: int
    n_control: int
    statistical_significance: bool
    p_value: float
    ci_lower: float
    ci_upper: float


def compute_naive_lift(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> LiftResult:
    """
    Compute unadjusted (naive) lift between treated and control groups.
    """
    treated = df[df[treatment_col] == 1][outcome_col].values
    control = df[df[treatment_col] == 0][outcome_col].values

    t_mean = float(np.mean(treated)) if len(treated) > 0 else 0.0
    c_mean = float(np.mean(control)) if len(control) > 0 else 0.0
    absolute_lift = t_mean - c_mean
    relative_lift = (absolute_lift / c_mean * 100) if c_mean != 0 else 0.0

    # Two-sample t-test
    if len(treated) > 1 and len(control) > 1:
        t_stat, p_val = stats.ttest_ind(treated, control, equal_var=False)
        se = absolute_lift / t_stat if t_stat != 0 else 0
    else:
        p_val = 1.0
        se = 0.0

    ci_lower = absolute_lift - 1.96 * abs(se)
    ci_upper = absolute_lift + 1.96 * abs(se)

    return LiftResult(
        absolute_lift=absolute_lift,
        relative_lift_pct=relative_lift,
        control_mean=c_mean,
        treatment_mean=t_mean,
        n_treated=len(treated),
        n_control=len(control),
        statistical_significance=p_val < 0.05,
        p_value=float(p_val),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compute_segment_lift(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    segment_col: str,
) -> pd.DataFrame:
    """
    Compute lift broken down by a segment variable (e.g., HCP_Specialty).
    """
    results = []
    for seg_val, seg_df in df.groupby(segment_col):
        if seg_df[treatment_col].nunique() < 2:
            continue
        lift = compute_naive_lift(seg_df, treatment_col, outcome_col)
        results.append({
            "Segment": seg_val,
            "N_Treated": lift.n_treated,
            "N_Control": lift.n_control,
            "Treatment_Mean": round(lift.treatment_mean, 4),
            "Control_Mean": round(lift.control_mean, 4),
            "Absolute_Lift": round(lift.absolute_lift, 4),
            "Relative_Lift_Pct": round(lift.relative_lift_pct, 2),
            "P_Value": round(lift.p_value, 4),
            "Significant": lift.statistical_significance,
        })
    return pd.DataFrame(results)


def compute_uplift_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    cate_predictions: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Compute cumulative uplift curve by targeting fraction.
    """
    order = np.argsort(-cate_predictions)
    sorted_outcome = outcome[order]
    sorted_treatment = treatment[order]

    n = len(outcome)
    fractions = []
    uplifts = []

    for i in range(1, n_bins + 1):
        k = int(n * i / n_bins)
        t_mask = sorted_treatment[:k] == 1
        c_mask = sorted_treatment[:k] == 0

        t_mean = np.mean(sorted_outcome[:k][t_mask]) if np.sum(t_mask) > 0 else 0
        c_mean = np.mean(sorted_outcome[:k][c_mask]) if np.sum(c_mask) > 0 else 0

        fractions.append(i / n_bins)
        uplifts.append(t_mean - c_mean)

    return pd.DataFrame({"Fraction_Targeted": fractions, "Cumulative_Uplift": uplifts})


def compute_qini_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    cate_predictions: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Compute Qini curve (incremental gains).
    """
    order = np.argsort(-cate_predictions)
    sorted_outcome = outcome[order]
    sorted_treatment = treatment[order]

    n = len(outcome)
    fractions = [0.0]
    qini_values = [0.0]

    for i in range(1, n_bins + 1):
        k = int(n * i / n_bins)
        t_mask = sorted_treatment[:k] == 1
        c_mask = sorted_treatment[:k] == 0

        n_t = np.sum(t_mask)
        n_c = np.sum(c_mask)

        if n_t > 0 and n_c > 0:
            incr = np.sum(sorted_outcome[:k][t_mask]) - np.sum(sorted_outcome[:k][c_mask]) * (n_t / n_c)
        else:
            incr = 0.0

        fractions.append(i / n_bins)
        qini_values.append(float(incr))

    return pd.DataFrame({"Fraction_Targeted": fractions, "Incremental_Gain": qini_values})


def compute_lift_by_decile(
    outcome: np.ndarray,
    treatment: np.ndarray,
    cate_predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Compute observed vs predicted lift by CATE decile.
    """
    n = len(outcome)
    decile_labels = pd.qcut(cate_predictions, q=10, labels=False, duplicates="drop")

    rows = []
    for d in sorted(np.unique(decile_labels)):
        mask = decile_labels == d
        t_mask = mask & (treatment == 1)
        c_mask = mask & (treatment == 0)

        t_mean = np.mean(outcome[t_mask]) if np.sum(t_mask) > 0 else 0
        c_mean = np.mean(outcome[c_mask]) if np.sum(c_mask) > 0 else 0
        observed_lift = t_mean - c_mean
        predicted_lift = np.mean(cate_predictions[mask])

        rows.append({
            "Decile": int(d) + 1,
            "N": int(np.sum(mask)),
            "N_Treated": int(np.sum(t_mask)),
            "N_Control": int(np.sum(c_mask)),
            "Observed_Lift": round(observed_lift, 4),
            "Predicted_Lift": round(predicted_lift, 4),
            "Mean_Outcome_Treated": round(t_mean, 4),
            "Mean_Outcome_Control": round(c_mean, 4),
        })

    return pd.DataFrame(rows)
