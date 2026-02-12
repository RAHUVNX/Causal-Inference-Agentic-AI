"""
Model Layer - Path A and Path B Regression Models

Two-path causal framework:
  Path A (Suggestion -> Action): predicts action_count from features
  Path B (Action -> Outcome): predicts outcome_count from features

Both use Ridge regression (OLS with L2 regularization).
"""

import numpy as np
from sklearn.linear_model import Ridge


class PathAActionModel:
    """Predicts action_count from suggestion features using Ridge regression."""

    def __init__(self):
        self.model = Ridge(alpha=1e-8, fit_intercept=True)
        self.feature_names: list[str] = []
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def train(
        self, features: list[list[float]], targets: list[float], feature_names: list[str]
    ) -> dict:
        self.feature_names = feature_names
        X = np.array(features)
        y = np.array(targets)

        self.model.fit(X, y)
        self.coefficients = self.model.coef_.tolist()
        self.intercept = float(self.model.intercept_)

        predictions = self.model.predict(X)
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
        return self.model.predict(X).tolist()

    def _compute_metrics(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> tuple[float, float]:
        n = len(actual)
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
        self.model = Ridge(alpha=1e-8, fit_intercept=True)
        self.feature_names: list[str] = []
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def train(
        self, features: list[list[float]], targets: list[float], feature_names: list[str]
    ) -> dict:
        self.feature_names = feature_names
        X = np.array(features)
        y = np.array(targets)

        self.model.fit(X, y)
        self.coefficients = self.model.coef_.tolist()
        self.intercept = float(self.model.intercept_)

        predictions = self.model.predict(X)
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
        return self.model.predict(X).tolist()

    def _compute_metrics(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> tuple[float, float]:
        n = len(actual)
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
