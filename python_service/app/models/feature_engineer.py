"""
Feature Engineering Pipeline

Mirrors the Spark FeatureEngineer component:
  1. Filter triangle: filter by suggestion_type, action_type, outcome_type
  2. Study period filtering
  3. Feature selection
  4. percent_other_hcps_suggested calculation
  5. Optional adstock transformation
  6. Categorical encoding (one-hot)
  7. Standardization (z-score scaling)
"""

import numpy as np
import pandas as pd
from typing import Optional

NUMERIC_FEATURE_COLS = [
    "suggestionCount",
    "actionCount",
    "tenureMonths",
    "priorTrx",
    "priorNbrx",
    "totalSuggestions",
    "percentOtherHcpsSuggested",
]

CATEGORICAL_COLS = ["specialtyCode", "regionCode"]

SUGGESTION_FEATURE_NAME = "suggestionCount"
PERCENT_OTHER_HCP_FEATURE_NAME = "percentOtherHcpsSuggested"


class FeatureEngineer:
    def __init__(
        self,
        suggestion_type: str,
        action_type: str,
        outcome_type: str,
        adstock_enabled: bool = False,
        adstock_decay: float = 0.5,
        theme: Optional[str] = None,
    ):
        self.suggestion_type = suggestion_type
        self.action_type = action_type
        self.outcome_type = outcome_type
        self.adstock_enabled = adstock_enabled
        self.adstock_decay = adstock_decay
        self.theme = theme
        self.scaler_means: dict[str, float] = {}
        self.scaler_stds: dict[str, float] = {}
        self.categorical_map: dict[str, dict[str, list[int]]] = {}
        self.feature_names: list[str] = []

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        # Step 1: Filter triangle
        filtered = df[
            (df["suggestionType"] == self.suggestion_type)
            & (df["actionType"] == self.action_type)
            & (df["outcomeType"] == self.outcome_type)
        ].copy()

        # Step 2: Theme filter
        if self.theme:
            filtered = filtered[
                filtered["theme"].str.lower() == self.theme.lower()
            ].copy()

        if len(filtered) == 0:
            raise ValueError(
                "No observations remain after filtering. Check filter triangle."
            )

        # Step 3: Compute percent_other_hcps_suggested per month
        filtered = self._compute_percent_other_hcps_suggested(filtered)

        # Step 4: Optional adstock transformation
        if self.adstock_enabled:
            filtered = self._apply_adstock(filtered, self.adstock_decay)

        # Step 5: Build categorical encoding map
        self.categorical_map = self._build_categorical_map(filtered)

        # Step 6 & 7: Standardize and assemble feature vectors
        result = self._standardize_and_assemble(filtered)

        return result

    def _compute_percent_other_hcps_suggested(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute what percentage of other HCPs also received a suggestion per month."""
        df = df.copy()
        df["monthKey"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)

        percent_values = []
        for _, row in df.iterrows():
            month_mask = df["monthKey"] == row["monthKey"]
            month_obs = df[month_mask]
            total_hcps = month_obs["hcpId"].nunique()
            suggested_hcps = month_obs[month_obs["suggestionCount"] > 0][
                "hcpId"
            ].nunique()

            is_suggested = 1 if row["suggestionCount"] > 0 else 0
            other_suggested = suggested_hcps - is_suggested
            other_total = total_hcps - 1

            pct = other_suggested / other_total if other_total > 0 else 0.0
            percent_values.append(pct)

        df["percentOtherHcpsSuggested"] = percent_values
        return df

    def _apply_adstock(self, df: pd.DataFrame, decay: float) -> pd.DataFrame:
        """Apply geometric decay to suggestion_count over time per HCP."""
        df = df.copy()
        df["monthDt"] = pd.to_datetime(df["month"])
        df = df.sort_values(["hcpId", "monthDt"])

        adstocked = []
        for _, group in df.groupby("hcpId"):
            prev = 0.0
            for idx in group.index:
                val = group.loc[idx, "suggestionCount"] + decay * prev
                adstocked.append((idx, val))
                prev = val

        for idx, val in adstocked:
            df.loc[idx, "suggestionCount"] = val

        if "monthDt" in df.columns:
            df = df.drop(columns=["monthDt"])
        return df

    def _build_categorical_map(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, list[int]]]:
        """Build one-hot encoding map for categorical columns."""
        cat_map: dict[str, dict[str, list[int]]] = {}
        for col in CATEGORICAL_COLS:
            unique_vals = sorted(
                df[col].dropna().unique().tolist()
            ) if col in df.columns else []
            unique_vals = [v for v in unique_vals if v != ""]
            cat_map[col] = {}
            for i, val in enumerate(unique_vals):
                one_hot = [0] * len(unique_vals)
                one_hot[i] = 1
                cat_map[col][val] = one_hot
        return cat_map

    def _standardize_and_assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize numeric features (z-score) and assemble feature vectors."""
        # Fill missing numeric values
        for col in NUMERIC_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Compute mean and std
        for col in NUMERIC_FEATURE_COLS:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            if std == 0:
                std = 1.0
            self.scaler_means[col] = float(mean)
            self.scaler_stds[col] = float(std)

        # Build feature names
        self.feature_names = list(NUMERIC_FEATURE_COLS)
        for col in CATEGORICAL_COLS:
            for cat in self.categorical_map.get(col, {}):
                self.feature_names.append(f"{col}_{cat}")

        # Assemble feature vectors
        features_list = []
        for _, row in df.iterrows():
            features = []
            # Standardized numeric features
            for col in NUMERIC_FEATURE_COLS:
                raw = float(row.get(col, 0) or 0)
                mean = self.scaler_means[col]
                std = self.scaler_stds[col]
                features.append((raw - mean) / std)

            # One-hot categorical features
            for col in CATEGORICAL_COLS:
                val = row.get(col)
                cats = self.categorical_map.get(col, {})
                if val and val in cats:
                    features.extend(cats[val])
                else:
                    features.extend([0] * len(cats))

            features_list.append(features)

        df = df.copy()
        df["features"] = features_list
        df["featureNames"] = [self.feature_names] * len(df)
        df["observedAction"] = df["actionCount"].values
        df["observedOutcome"] = df["outcomeCount"].values

        return df
