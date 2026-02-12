"""
Model Layer - Path A and Path B Regression Models

Two-path causal framework:
  Path A (Suggestion -> Action): predicts action_count from features
  Path B (Action -> Outcome): predicts outcome_count from features

Both use Ridge regression (OLS with L2 regularization) implemented
via the normal equation: beta = (X'X + alpha*I)^{-1} X'y
No scikit-learn dependency required.
"""

import numpy as np


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-8):
    """
    Fit Ridge regression via normal equation.
    Returns (coefficients, intercept).
    """
    n, p = X.shape
    # Add intercept column
    X_bias = np.column_stack([np.ones(n), X])

    # Normal equation with ridge penalty (skip intercept column for penalty)
    XtX = X_bias.T @ X_bias
    penalty = alpha * np.eye(p + 1)
    penalty[0, 0] = 0  # Don't penalize the intercept
    XtX += penalty

    Xty = X_bias.T @ y
    beta = np.linalg.solve(XtX, Xty)

    intercept = float(beta[0])
    coefficients = beta[1:].tolist()
    return coefficients, intercept


def _ridge_predict(X: np.ndarray, coefficients: list[float], intercept: float) -> np.ndarray:
    """Predict using Ridge regression coefficients."""
    return X @ np.array(coefficients) + intercept


class PathAActionModel:
    """Predicts action_count from suggestion features using Ridge regression."""

    def __init__(self):
        self.feature_names: list[str] = []
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def train(
        self, features: list[list[float]], targets: list[float], feature_names: list[str]
    ) -> dict:
        self.feature_names = feature_names
        X = np.array(features)
        y = np.array(targets)

        self.coefficients, self.intercept = _ridge_fit(X, y)

        predictions = _ridge_predict(X, self.coefficients, self.intercept)
        r_squared, mae = self._compute_metrics(y, predictions)
        importance = self._extract_feature_importance()

        return {
            "modelName": "PathA_Action",
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "featureNames": self.feature_names,
            "featureImportance": importance,
            "rSquared": r_squared,
            "mae": mae,
        }

    def predict(self, features: list[list[float]]) -> list[float]:
        X = np.array(features)
        return _ridge_predict(X, self.coefficients, self.intercept).tolist()

    def _compute_metrics(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> tuple[float, float]:
        mean = actual.mean()
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - mean) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(actual - predicted)))
        return r_squared, mae

    def _extract_feature_importance(self) -> list[dict]:
        entries = [
            {
                "featureName": name,
                "importance": abs(coef),
                "rank": 0,
            }
            for name, coef in zip(self.feature_names, self.coefficients)
        ]
        entries.sort(key=lambda x: x["importance"], reverse=True)
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
        return entries


class PathBOutcomeModel:
    """Predicts outcome_count from action and other features using Ridge regression."""

    def __init__(self, outcome_type: str):
        self.outcome_type = outcome_type
        self.feature_names: list[str] = []
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def train(
        self, features: list[list[float]], targets: list[float], feature_names: list[str]
    ) -> dict:
        self.feature_names = feature_names
        X = np.array(features)
        y = np.array(targets)

        self.coefficients, self.intercept = _ridge_fit(X, y)

        predictions = _ridge_predict(X, self.coefficients, self.intercept)
        r_squared, mae = self._compute_metrics(y, predictions)
        importance = self._extract_feature_importance()

        model_name = "PathB_TRX" if self.outcome_type == "trx" else "PathB_NBRX"

        return {
            "modelName": model_name,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "featureNames": self.feature_names,
            "featureImportance": importance,
            "rSquared": r_squared,
            "mae": mae,
        }

    def predict(self, features: list[list[float]]) -> list[float]:
        X = np.array(features)
        return _ridge_predict(X, self.coefficients, self.intercept).tolist()

    def _compute_metrics(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> tuple[float, float]:
        mean = actual.mean()
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - mean) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(actual - predicted)))
        return r_squared, mae

    def _extract_feature_importance(self) -> list[dict]:
        entries = [
            {
                "featureName": name,
                "importance": abs(coef),
                "rank": 0,
            }
            for name, coef in zip(self.feature_names, self.coefficients)
        ]
        entries.sort(key=lambda x: x["importance"], reverse=True)
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
        return entries
