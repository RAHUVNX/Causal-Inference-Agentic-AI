"""Configuration objects for the two-stage causal modeling pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple


class ConfigError(ValueError):
    """Raised when a ModelConfig fails validation."""


def _as_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class StageConfig:
    """Stage configuration for a single causal path.

    The stage is defined by a source table (features) and a target table
    (label), with explicit timestamps and identifiers to support strict
    temporal ordering.
    """

    name: str
    source_table: str
    target_table: str
    source_id_col: str
    target_id_col: str
    source_ts_col: str
    target_ts_col: str
    label_col: str
    features: Tuple[str, ...] = field(default_factory=tuple)
    categorical_features: Tuple[str, ...] = field(default_factory=tuple)
    aggregation_keys: Tuple[str, ...] = field(default_factory=tuple)
    filters: Tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("StageConfig.name is required.")
        if not self.source_table or not self.target_table:
            raise ConfigError(f"{self.name}: source_table and target_table are required.")
        if not self.source_id_col or not self.target_id_col:
            raise ConfigError(f"{self.name}: source_id_col and target_id_col are required.")
        if not self.source_ts_col or not self.target_ts_col:
            raise ConfigError(f"{self.name}: source_ts_col and target_ts_col are required.")
        if not self.label_col:
            raise ConfigError(f"{self.name}: label_col is required.")

        if "all" in self.features or "all" in self.categorical_features:
            raise ConfigError(
                f"{self.name}: 'all' is an aggregation instruction and cannot be a feature."
            )
        if self.label_col in self.features:
            raise ConfigError(f"{self.name}: label_col must not appear in features.")
        if len(set(self.features)) != len(self.features):
            raise ConfigError(f"{self.name}: features must be unique.")
        if len(set(self.categorical_features)) != len(self.categorical_features):
            raise ConfigError(f"{self.name}: categorical_features must be unique.")
        if self.label_col in self.categorical_features:
            raise ConfigError(f"{self.name}: label_col must not be categorical.")

        overlap = set(self.features).intersection(self.categorical_features)
        if overlap:
            raise ConfigError(
                f"{self.name}: features and categorical_features overlap: {sorted(overlap)}."
            )
        if len(set(self.aggregation_keys)) != len(self.aggregation_keys):
            raise ConfigError(f"{self.name}: aggregation_keys must be unique.")
        if "all" in self.aggregation_keys and len(self.aggregation_keys) > 1:
            raise ConfigError(
                f"{self.name}: 'all' must be the only aggregation key when specified."
            )
        if not all(self.filters):
            raise ConfigError(f"{self.name}: filters cannot contain empty strings.")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the two-stage causal pipeline.

    Path A: Suggestions -> Actions
    Path B: Actions -> Outcomes
    """

    path_a: StageConfig
    path_b: StageConfig
    experiment_id_col: str
    unit_id_col: str
    timezone: str = "UTC"
    random_seed: int = 0

    def validate(self) -> None:
        if not self.experiment_id_col:
            raise ConfigError("experiment_id_col is required.")
        if not self.unit_id_col:
            raise ConfigError("unit_id_col is required.")

        self.path_a.validate()
        self.path_b.validate()

        if self.path_a.name == self.path_b.name:
            raise ConfigError("path_a and path_b must have distinct names.")
        if self.path_a.target_table != self.path_b.source_table:
            raise ConfigError(
                "path_a.target_table must feed into path_b.source_table to preserve ordering."
            )
        if self.path_a.label_col == self.path_b.label_col:
            raise ConfigError("path_a.label_col and path_b.label_col must be distinct.")

        leakage = set(self.path_b.features).intersection(self.path_a.features)
        if leakage:
            raise ConfigError(
                "path_a and path_b features must be disjoint to avoid leakage: "
                f"{sorted(leakage)}."
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ModelConfig":
        path_a = StageConfig(
            name=str(payload["path_a"]["name"]),
            source_table=str(payload["path_a"]["source_table"]),
            target_table=str(payload["path_a"]["target_table"]),
            source_id_col=str(payload["path_a"]["source_id_col"]),
            target_id_col=str(payload["path_a"]["target_id_col"]),
            source_ts_col=str(payload["path_a"]["source_ts_col"]),
            target_ts_col=str(payload["path_a"]["target_ts_col"]),
            label_col=str(payload["path_a"]["label_col"]),
            features=_as_tuple(payload["path_a"].get("features", ())),
            categorical_features=_as_tuple(payload["path_a"].get("categorical_features", ())),
            aggregation_keys=_as_tuple(payload["path_a"].get("aggregation_keys", ())),
            filters=_as_tuple(payload["path_a"].get("filters", ())),
        )
        path_b = StageConfig(
            name=str(payload["path_b"]["name"]),
            source_table=str(payload["path_b"]["source_table"]),
            target_table=str(payload["path_b"]["target_table"]),
            source_id_col=str(payload["path_b"]["source_id_col"]),
            target_id_col=str(payload["path_b"]["target_id_col"]),
            source_ts_col=str(payload["path_b"]["source_ts_col"]),
            target_ts_col=str(payload["path_b"]["target_ts_col"]),
            label_col=str(payload["path_b"]["label_col"]),
            features=_as_tuple(payload["path_b"].get("features", ())),
            categorical_features=_as_tuple(payload["path_b"].get("categorical_features", ())),
            aggregation_keys=_as_tuple(payload["path_b"].get("aggregation_keys", ())),
            filters=_as_tuple(payload["path_b"].get("filters", ())),
        )
        config = cls(
            path_a=path_a,
            path_b=path_b,
            experiment_id_col=str(payload["experiment_id_col"]),
            unit_id_col=str(payload["unit_id_col"]),
            timezone=str(payload.get("timezone", "UTC")),
            random_seed=int(payload.get("random_seed", 0)),
        )
        config.validate()
        return config
