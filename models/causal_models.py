"""
Causal inference model implementations.

Provides multiple estimation strategies:
  1. Propensity Score Matching (PSM)
  2. Inverse Propensity Weighting (IPW)
  3. Doubly Robust Estimation
  4. Difference-in-Differences (DID) style before/after comparison
  5. Meta-Learners (S-Learner, T-Learner) for heterogeneous treatment effects
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class CausalEstimate:
    """Container for a causal effect estimate."""

    method: str
    ate: float  # Average Treatment Effect
    ate_se: float  # Standard error
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    p_value: float
    n_treated: int
    n_control: int
    details: dict


# ---------------------------------------------------------------------------
# Propensity Score helpers
# ---------------------------------------------------------------------------

def estimate_propensity_scores(
    X: np.ndarray,
    treatment: np.ndarray,
    model_type: str = "logistic",
) -> np.ndarray:
    """Estimate propensity scores P(T=1|X)."""
    if model_type == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)

    scores = cross_val_predict(clf, X, treatment, cv=5, method="predict_proba")[:, 1]
    return scores.clip(0.01, 0.99)


# ---------------------------------------------------------------------------
# 1. Propensity Score Matching
# ---------------------------------------------------------------------------

def propensity_score_matching(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    outcome_col: str,
    n_neighbors: int = 1,
    caliper: Optional[float] = 0.05,
    ps_model: str = "logistic",
) -> CausalEstimate:
    """Estimate ATE via nearest-neighbor propensity score matching."""
    X = df[covariates].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    T = df[treatment_col].values
    Y = df[outcome_col].values

    ps = estimate_propensity_scores(X_scaled, T, model_type=ps_model)

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

    # Apply caliper
    if caliper is not None:
        mask = distances[:, 0] <= caliper
    else:
        mask = np.ones(len(treated_idx), dtype=bool)

    matched_treated = treated_idx[mask]
    matched_control = control_idx[indices[mask, 0]]

    if len(matched_treated) == 0:
        return CausalEstimate(
            method="Propensity Score Matching",
            ate=0.0, ate_se=0.0,
            ci_lower=0.0, ci_upper=0.0,
            p_value=1.0,
            n_treated=0, n_control=0,
            details={"error": "No matches found within caliper"},
        )

    diff = Y[matched_treated] - Y[matched_control]
    ate = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="Propensity Score Matching",
        ate=round(ate, 6),
        ate_se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_treated=int(len(matched_treated)),
        n_control=int(len(matched_control)),
        details={
            "n_matched": int(mask.sum()),
            "n_unmatched": int((~mask).sum()),
            "caliper": caliper,
            "propensity_scores": ps,
            "matched_treated_idx": matched_treated,
            "matched_control_idx": matched_control,
        },
    )


# ---------------------------------------------------------------------------
# 2. Inverse Propensity Weighting (IPW)
# ---------------------------------------------------------------------------

def inverse_propensity_weighting(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    outcome_col: str,
    ps_model: str = "logistic",
) -> CausalEstimate:
    """Estimate ATE using Horvitz-Thompson IPW estimator."""
    X = df[covariates].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    T = df[treatment_col].values.astype(float)
    Y = df[outcome_col].values.astype(float)

    ps = estimate_propensity_scores(X_scaled, T.astype(int), model_type=ps_model)

    # IPW estimator
    w1 = T / ps
    w0 = (1 - T) / (1 - ps)
    ate = float(np.mean(w1 * Y) - np.mean(w0 * Y))

    # Bootstrap standard error
    n_boot = 500
    rng = np.random.RandomState(42)
    boot_ates = []
    n = len(Y)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        b_ate = np.mean(w1[idx] * Y[idx]) - np.mean(w0[idx] * Y[idx])
        boot_ates.append(b_ate)
    se = float(np.std(boot_ates, ddof=1))

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="Inverse Propensity Weighting",
        ate=round(ate, 6),
        ate_se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_treated=int(T.sum()),
        n_control=int((1 - T).sum()),
        details={"propensity_scores": ps, "weights_treated": w1, "weights_control": w0},
    )


# ---------------------------------------------------------------------------
# 3. Doubly Robust Estimation (AIPW)
# ---------------------------------------------------------------------------

def doubly_robust(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    outcome_col: str,
    ps_model: str = "logistic",
) -> CausalEstimate:
    """Augmented IPW — consistent if either the propensity or outcome model is correct."""
    X = df[covariates].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    T = df[treatment_col].values.astype(float)
    Y = df[outcome_col].values.astype(float)

    ps = estimate_propensity_scores(X_scaled, T.astype(int), model_type=ps_model)

    # Outcome models
    treated_mask = T == 1
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

    reg.fit(X_scaled[treated_mask], Y[treated_mask])
    mu1 = reg.predict(X_scaled)

    reg.fit(X_scaled[~treated_mask], Y[~treated_mask])
    mu0 = reg.predict(X_scaled)

    # AIPW score
    dr1 = mu1 + T * (Y - mu1) / ps
    dr0 = mu0 + (1 - T) * (Y - mu0) / (1 - ps)
    scores = dr1 - dr0
    ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="Doubly Robust (AIPW)",
        ate=round(ate, 6),
        ate_se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_treated=int(T.sum()),
        n_control=int((1 - T).sum()),
        details={
            "propensity_scores": ps,
            "mu1_pred": mu1,
            "mu0_pred": mu0,
            "individual_scores": scores,
        },
    )


# ---------------------------------------------------------------------------
# 4. S-Learner (single model for heterogeneous treatment effects)
# ---------------------------------------------------------------------------

def s_learner(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    outcome_col: str,
) -> CausalEstimate:
    """S-Learner: fit one model with treatment as a feature, predict CATE."""
    X = df[covariates].values
    T = df[treatment_col].values.reshape(-1, 1)
    Y = df[outcome_col].values

    X_with_t = np.hstack([X, T])
    scaler = StandardScaler()
    X_with_t_scaled = scaler.fit_transform(X_with_t)

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    reg.fit(X_with_t_scaled, Y)

    X1 = scaler.transform(np.hstack([X, np.ones((len(X), 1))]))
    X0 = scaler.transform(np.hstack([X, np.zeros((len(X), 1))]))

    cate = reg.predict(X1) - reg.predict(X0)
    ate = float(np.mean(cate))
    se = float(np.std(cate, ddof=1) / np.sqrt(len(cate)))

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="S-Learner",
        ate=round(ate, 6),
        ate_se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_treated=int(df[treatment_col].sum()),
        n_control=int((1 - df[treatment_col]).sum()),
        details={"cate": cate},
    )


# ---------------------------------------------------------------------------
# 5. T-Learner (two separate models for heterogeneous treatment effects)
# ---------------------------------------------------------------------------

def t_learner(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    outcome_col: str,
) -> CausalEstimate:
    """T-Learner: fit separate models for treated and control, predict CATE."""
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    treated_mask = T == 1

    reg1 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    reg1.fit(X_scaled[treated_mask], Y[treated_mask])

    reg0 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    reg0.fit(X_scaled[~treated_mask], Y[~treated_mask])

    cate = reg1.predict(X_scaled) - reg0.predict(X_scaled)
    ate = float(np.mean(cate))
    se = float(np.std(cate, ddof=1) / np.sqrt(len(cate)))

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="T-Learner",
        ate=round(ate, 6),
        ate_se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_treated=int(T.sum()),
        n_control=int((1 - T).sum()),
        details={"cate": cate},
    )
