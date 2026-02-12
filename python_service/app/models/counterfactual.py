"""
Counterfactual Simulation Engine

Core causal inference logic:
1. Zero out suggestion features -> counterfactual_action_count
2. Replace action feature with predicted/counterfactual action -> outcome predictions
3. Compute incremental lift as the difference
"""

import numpy as np
from .feature_engineer import (
    SUGGESTION_FEATURE_NAME,
    PERCENT_OTHER_HCP_FEATURE_NAME,
)
from .regression import PathAActionModel, PathBOutcomeModel

SUGGESTION_RELATED_FEATURES = [
    SUGGESTION_FEATURE_NAME,
    PERCENT_OTHER_HCP_FEATURE_NAME,
    "totalSuggestions",
]


class CounterfactualEngine:
    def __init__(
        self,
        path_a_model: PathAActionModel,
        path_b_model: PathBOutcomeModel,
        scaler_means: dict[str, float],
        scaler_stds: dict[str, float],
    ):
        self.path_a_model = path_a_model
        self.path_b_model = path_b_model
        self.scaler_means = scaler_means
        self.scaler_stds = scaler_stds

    def simulate(
        self,
        features: list[list[float]],
        feature_names: list[str],
        rows_meta: list[dict],
        config: dict,
    ) -> list[dict]:
        """
        Run full counterfactual simulation.

        For each observation:
          PATH A:
            predicted_action      = PathA.predict(original_features)
            counterfactual_action = PathA.predict(zeroed_suggestion_features)

          PATH B:
            predicted_outcome      = PathB.predict(features_with_predicted_action)
            counterfactual_outcome = PathB.predict(features_with_counterfactual_action)
        """
        # Step 1: Path A - Predict action with original features
        predicted_actions = self.path_a_model.predict(features)

        # Step 2: Zero out suggestion features for counterfactual
        cf_features = self._zero_out_suggestion_features(features, feature_names)

        # Step 3: Path A - Predict counterfactual action
        counterfactual_actions = self.path_a_model.predict(cf_features)

        # Step 4: Path B - Predict outcome with predicted action
        features_with_predicted = self._substitute_action_feature(
            features, feature_names, predicted_actions
        )
        predicted_outcomes = self.path_b_model.predict(features_with_predicted)

        # Step 5: Path B - Predict outcome with counterfactual action
        features_with_counterfactual = self._substitute_action_feature(
            features, feature_names, counterfactual_actions
        )
        counterfactual_outcomes = self.path_b_model.predict(
            features_with_counterfactual
        )

        # Step 6: Compute incremental lift
        results = []
        for i, meta in enumerate(rows_meta):
            incremental_action = predicted_actions[i] - counterfactual_actions[i]
            incremental_outcome = predicted_outcomes[i] - counterfactual_outcomes[i]

            results.append(
                {
                    "hcpId": meta["hcpId"],
                    "month": meta["month"],
                    "suggestionType": config["suggestionType"],
                    "actionType": config["actionType"],
                    "outcomeType": config["outcomeType"],
                    "theme": meta.get("theme"),
                    "predictedAction": predicted_actions[i],
                    "counterfactualAction": counterfactual_actions[i],
                    "incrementalAction": incremental_action,
                    "predictedOutcome": predicted_outcomes[i],
                    "counterfactualOutcome": counterfactual_outcomes[i],
                    "incrementalOutcome": incremental_outcome,
                    "observedAction": meta["observedAction"],
                    "observedOutcome": meta["observedOutcome"],
                }
            )

        return results

    def _zero_out_suggestion_features(
        self, features: list[list[float]], feature_names: list[str]
    ) -> list[list[float]]:
        """Zero out suggestion-related features using standardized zero values."""
        suggestion_indices = [
            i
            for i, name in enumerate(feature_names)
            if name in SUGGESTION_RELATED_FEATURES
        ]

        result = []
        for row in features:
            new_row = list(row)
            for idx in suggestion_indices:
                name = feature_names[idx]
                mean = self.scaler_means.get(name, 0)
                std = self.scaler_stds.get(name, 1)
                new_row[idx] = (0 - mean) / std
            result.append(new_row)
        return result

    def _substitute_action_feature(
        self,
        features: list[list[float]],
        feature_names: list[str],
        substitute_actions: list[float],
    ) -> list[list[float]]:
        """Replace actionCount feature with substitute values (standardized)."""
        try:
            action_idx = feature_names.index("actionCount")
        except ValueError:
            return [list(row) for row in features]

        mean = self.scaler_means.get("actionCount", 0)
        std = self.scaler_stds.get("actionCount", 1)

        result = []
        for i, row in enumerate(features):
            new_row = list(row)
            new_row[action_idx] = (substitute_actions[i] - mean) / std
            result.append(new_row)
        return result
