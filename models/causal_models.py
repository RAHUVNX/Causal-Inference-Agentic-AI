"""
Causal inference models for Rep-HCP outreach analysis.

Two-stage estimation:
  Stage 1 - Incremental Actions: Effect of suggestions on rep actions
  Stage 2 - Incremental Outcomes: Effect of actions on TRX / NBRX
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


@dataclass
class CausalEstimate:
    """Container for a single causal effect estimate."""
    method: str
    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    details: dict = field(default_factory=dict)


@dataclass
class CATEResult:
    """Container for conditional average treatment effect predictions."""
    method: str
    cate_predictions: np.ndarray
    ate: float
    feature_importances: Optional[dict] = None


def encode_features(df: pd.DataFrame, covariate_cols: list) -> np.ndarray:
    """Encode covariates (including categoricals) into numeric matrix."""
    encoded_parts = []
    for col in covariate_cols:
        if df[col].dtype == object or df[col].dtype.name == "category":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(float)
            encoded_parts.append(dummies.values)
        else:
            encoded_parts.append(df[[col]].values.astype(float))
    return np.hstack(encoded_parts)


def estimate_propensity_scores(
    X: np.ndarray,
    treatment: np.ndarray,
    model_type: str = "logistic",
) -> np.ndarray:
    """
    Estimate propensity scores using cross-validated predictions.

    Parameters
    ----------
    X : array-like
        Covariate matrix.
    treatment : array-like
        Binary treatment indicator.
    model_type : str
        'logistic' or 'gbm'.

    Returns
    -------
    np.ndarray
        Propensity score estimates.
    """
    if model_type == "gbm":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    ps = cross_val_predict(model, X, treatment, cv=5, method="predict_proba")[:, 1]
    return np.clip(ps, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Method 1: Propensity Score Matching
# ---------------------------------------------------------------------------

def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
    ps_model: str = "logistic",
    caliper: float = 0.1,
) -> CausalEstimate:
    """
    Estimate ATE via nearest-neighbor propensity score matching.
    """
    X = encode_features(df, covariate_cols)
    treatment = df[treatment_col].values
    outcome = df[outcome_col].values

    ps = estimate_propensity_scores(X, treatment, ps_model)

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

    # Apply caliper
    valid = distances.ravel() <= caliper
    matched_treated = treated_idx[valid]
    matched_control = control_idx[indices.ravel()[valid]]

    if len(matched_treated) == 0:
        return CausalEstimate(
            method="Propensity Score Matching",
            ate=0.0, ate_se=0.0, ci_lower=0.0, ci_upper=0.0,
            p_value=1.0, n_treated=0, n_control=0,
            details={"matched_pairs": 0, "caliper": caliper},
        )

    diffs = outcome[matched_treated] - outcome[matched_control]
    ate = float(np.mean(diffs))
    ate_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    ci_lower = ate - 1.96 * ate_se
    ci_upper = ate + 1.96 * ate_se
    t_stat = ate / ate_se if ate_se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="Propensity Score Matching",
        ate=ate, ate_se=ate_se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        details={
            "matched_pairs": int(len(matched_treated)),
            "caliper": caliper,
            "mean_distance": float(np.mean(distances[valid])),
        },
    )


# ---------------------------------------------------------------------------
# Method 2: Inverse Propensity Weighting (IPW)
# ---------------------------------------------------------------------------

def inverse_propensity_weighting(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
    ps_model: str = "logistic",
    n_bootstrap: int = 500,
) -> CausalEstimate:
    """
    Estimate ATE using Horvitz-Thompson IPW estimator with bootstrap SEs.
    """
    X = encode_features(df, covariate_cols)
    treatment = df[treatment_col].values.astype(float)
    outcome = df[outcome_col].values.astype(float)

    ps = estimate_propensity_scores(X, treatment, ps_model)

    # Horvitz-Thompson estimator
    w1 = treatment / ps
    w0 = (1 - treatment) / (1 - ps)
    ate = float(np.mean(w1 * outcome) - np.mean(w0 * outcome))

    # Bootstrap standard error
    rng = np.random.default_rng(42)
    n = len(outcome)
    boot_ates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_t, b_y, b_ps = treatment[idx], outcome[idx], ps[idx]
        bw1 = b_t / b_ps
        bw0 = (1 - b_t) / (1 - b_ps)
        boot_ates.append(np.mean(bw1 * b_y) - np.mean(bw0 * b_y))

    ate_se = float(np.std(boot_ates, ddof=1))
    ci_lower = float(np.percentile(boot_ates, 2.5))
    ci_upper = float(np.percentile(boot_ates, 97.5))
    t_stat = ate / ate_se if ate_se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="Inverse Propensity Weighting",
        ate=ate, ate_se=ate_se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        details={"n_bootstrap": n_bootstrap},
    )


# ---------------------------------------------------------------------------
# Method 3: Doubly Robust (AIPW)
# ---------------------------------------------------------------------------

def doubly_robust(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
    ps_model: str = "logistic",
) -> CausalEstimate:
    """
    Augmented Inverse Propensity Weighting (doubly robust estimator).
    """
    X = encode_features(df, covariate_cols)
    treatment = df[treatment_col].values.astype(float)
    outcome = df[outcome_col].values.astype(float)

    ps = estimate_propensity_scores(X, treatment, ps_model)

    # Outcome models
    treated_mask = treatment == 1
    control_mask = treatment == 0

    mu1_model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
    )
    mu0_model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
    )

    if np.sum(treated_mask) > 5 and np.sum(control_mask) > 5:
        mu1_model.fit(X[treated_mask], outcome[treated_mask])
        mu0_model.fit(X[control_mask], outcome[control_mask])
        mu1 = mu1_model.predict(X)
        mu0 = mu0_model.predict(X)
    else:
        mu1 = np.full(len(outcome), np.mean(outcome[treated_mask]) if np.any(treated_mask) else 0)
        mu0 = np.full(len(outcome), np.mean(outcome[control_mask]) if np.any(control_mask) else 0)

    # AIPW scores
    scores = (
        mu1 - mu0
        + treatment * (outcome - mu1) / ps
        - (1 - treatment) * (outcome - mu0) / (1 - ps)
    )

    ate = float(np.mean(scores))
    ate_se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    ci_lower = ate - 1.96 * ate_se
    ci_upper = ate + 1.96 * ate_se
    t_stat = ate / ate_se if ate_se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    return CausalEstimate(
        method="Doubly Robust (AIPW)",
        ate=ate, ate_se=ate_se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        details={},
    )


# ---------------------------------------------------------------------------
# Method 4: S-Learner
# ---------------------------------------------------------------------------

def s_learner(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
) -> tuple[CausalEstimate, CATEResult]:
    """
    S-Learner: single model with treatment as a feature.
    Returns both ATE and individual CATE predictions.
    """
    X = encode_features(df, covariate_cols)
    treatment = df[treatment_col].values.astype(float)
    outcome = df[outcome_col].values.astype(float)

    # Add treatment as feature
    X_with_t = np.column_stack([X, treatment])

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(X_with_t, outcome)

    # Predict under treatment=1 and treatment=0
    X_t1 = np.column_stack([X, np.ones(len(X))])
    X_t0 = np.column_stack([X, np.zeros(len(X))])

    pred_t1 = model.predict(X_t1)
    pred_t0 = model.predict(X_t0)
    cate = pred_t1 - pred_t0

    ate = float(np.mean(cate))

    # Bootstrap SE
    rng = np.random.default_rng(42)
    n = len(cate)
    boot_ates = [np.mean(cate[rng.choice(n, size=n, replace=True)]) for _ in range(300)]
    ate_se = float(np.std(boot_ates, ddof=1))
    ci_lower = float(np.percentile(boot_ates, 2.5))
    ci_upper = float(np.percentile(boot_ates, 97.5))
    t_stat = ate / ate_se if ate_se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    estimate = CausalEstimate(
        method="S-Learner",
        ate=ate, ate_se=ate_se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        details={},
    )

    cate_result = CATEResult(
        method="S-Learner",
        cate_predictions=cate,
        ate=ate,
    )

    return estimate, cate_result


# ---------------------------------------------------------------------------
# Method 5: T-Learner
# ---------------------------------------------------------------------------

def t_learner(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
) -> tuple[CausalEstimate, CATEResult]:
    """
    T-Learner: separate models for treated and control groups.
    Returns both ATE and individual CATE predictions.
    """
    X = encode_features(df, covariate_cols)
    treatment = df[treatment_col].values.astype(float)
    outcome = df[outcome_col].values.astype(float)

    treated_mask = treatment == 1
    control_mask = treatment == 0

    model_t = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    model_c = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )

    model_t.fit(X[treated_mask], outcome[treated_mask])
    model_c.fit(X[control_mask], outcome[control_mask])

    pred_t = model_t.predict(X)
    pred_c = model_c.predict(X)
    cate = pred_t - pred_c

    ate = float(np.mean(cate))

    rng = np.random.default_rng(42)
    n = len(cate)
    boot_ates = [np.mean(cate[rng.choice(n, size=n, replace=True)]) for _ in range(300)]
    ate_se = float(np.std(boot_ates, ddof=1))
    ci_lower = float(np.percentile(boot_ates, 2.5))
    ci_upper = float(np.percentile(boot_ates, 97.5))
    t_stat = ate / ate_se if ate_se > 0 else 0
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    estimate = CausalEstimate(
        method="T-Learner",
        ate=ate, ate_se=ate_se, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        details={},
    )

    cate_result = CATEResult(
        method="T-Learner",
        cate_predictions=cate,
        ate=ate,
    )

    return estimate, cate_result
