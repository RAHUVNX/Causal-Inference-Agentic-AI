"""
Causal Lift Python ML Service

FastAPI application providing ML endpoints:
  POST /train       - Train Path A and Path B models
  POST /run-lift    - Full pipeline: train + counterfactual simulation
  GET  /health      - Health check
"""

from collections import defaultdict
from fastapi import FastAPI, HTTPException
import pandas as pd

from .schemas import (
    TrainRequest,
    TrainResponse,
    TrainedModelResult,
    FeatureImportanceEntry,
    LiftRequest,
    LiftResponse,
    LiftResultRow,
    MonthlySummary,
    ModelMetrics,
)
from .models.feature_engineer import FeatureEngineer
from .models.regression import PathAActionModel, PathBOutcomeModel
from .models.counterfactual import CounterfactualEngine
from .models.metrics import compute_full_metrics

app = FastAPI(title="Causal Lift ML Service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "python-ml"}


def _observations_to_dataframe(observations: list) -> pd.DataFrame:
    """Convert list of ObservationRow pydantic models to a pandas DataFrame."""
    records = [obs.model_dump() for obs in observations]
    return pd.DataFrame(records)


def _run_feature_engineering(req) -> tuple[FeatureEngineer, pd.DataFrame]:
    """Run feature engineering and return the engineer and engineered DataFrame."""
    df = _observations_to_dataframe(req.observations)

    fe = FeatureEngineer(
        suggestion_type=req.filter.suggestionType,
        action_type=req.filter.actionType,
        outcome_type=req.filter.outcomeType,
        adstock_enabled=req.adstockEnabled,
        adstock_decay=req.adstockDecay,
        theme=req.theme,
    )

    engineered = fe.engineer(df)
    return fe, engineered


def _build_model_result(
    model_result: dict,
    scaler_means: dict,
    scaler_stds: dict,
    categorical_map: dict,
) -> TrainedModelResult:
    """Build a TrainedModelResult from raw model training output."""
    return TrainedModelResult(
        modelName=model_result["modelName"],
        coefficients=model_result["coefficients"],
        intercept=model_result["intercept"],
        featureNames=model_result["featureNames"],
        scalerMeans=scaler_means,
        scalerStds=scaler_stds,
        categoricalMap=categorical_map,
        featureImportance=[
            FeatureImportanceEntry(**fi) for fi in model_result["featureImportance"]
        ],
        rSquared=model_result["rSquared"],
        mae=model_result["mae"],
    )


@app.post("/train", response_model=TrainResponse)
def train_models(req: TrainRequest):
    """Train Path A and Path B models."""
    try:
        fe, engineered = _run_feature_engineering(req)

        features = engineered["features"].tolist()
        action_targets = engineered["actionCount"].tolist()
        outcome_targets = engineered["outcomeCount"].tolist()
        feature_names = fe.feature_names

        # Train Path A
        path_a = PathAActionModel()
        path_a_result = path_a.train(features, action_targets, feature_names)

        # Train Path B
        path_b = PathBOutcomeModel(req.filter.outcomeType)
        path_b_result = path_b.train(features, outcome_targets, feature_names)

        return TrainResponse(
            pathA=_build_model_result(
                path_a_result, fe.scaler_means, fe.scaler_stds, fe.categorical_map
            ),
            pathB=_build_model_result(
                path_b_result, fe.scaler_means, fe.scaler_stds, fe.categorical_map
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/run-lift", response_model=LiftResponse)
def run_lift(req: LiftRequest):
    """Execute full pipeline: feature engineering + train + counterfactual simulation."""
    try:
        fe, engineered = _run_feature_engineering(req)

        features = engineered["features"].tolist()
        action_targets = engineered["actionCount"].tolist()
        outcome_targets = engineered["outcomeCount"].tolist()
        feature_names = fe.feature_names

        # Train Path A
        path_a = PathAActionModel()
        path_a_result = path_a.train(features, action_targets, feature_names)

        # Train Path B
        path_b = PathBOutcomeModel(req.filter.outcomeType)
        path_b_result = path_b.train(features, outcome_targets, feature_names)

        # Build row metadata for counterfactual results
        rows_meta = []
        for _, row in engineered.iterrows():
            rows_meta.append(
                {
                    "hcpId": row["hcpId"],
                    "month": row["month"],
                    "theme": row.get("theme"),
                    "observedAction": float(row["observedAction"]),
                    "observedOutcome": float(row["observedOutcome"]),
                }
            )

        # Run counterfactual simulation
        cf_engine = CounterfactualEngine(
            path_a_model=path_a,
            path_b_model=path_b,
            scaler_means=fe.scaler_means,
            scaler_stds=fe.scaler_stds,
        )

        lift_results = cf_engine.simulate(
            features=features,
            feature_names=feature_names,
            rows_meta=rows_meta,
            config={
                "suggestionType": req.filter.suggestionType,
                "actionType": req.filter.actionType,
                "outcomeType": req.filter.outcomeType,
            },
        )

        # Compute metrics
        metrics = compute_full_metrics(lift_results)

        # Aggregate monthly summaries
        monthly_summary = _aggregate_monthly_summary(lift_results)

        return LiftResponse(
            liftResults=[LiftResultRow(**lr) for lr in lift_results],
            monthlySummary=monthly_summary,
            metrics={
                "pathA": ModelMetrics(**metrics["pathA"]),
                "pathB": ModelMetrics(**metrics["pathB"]),
            },
            pathA=_build_model_result(
                path_a_result, fe.scaler_means, fe.scaler_stds, fe.categorical_map
            ),
            pathB=_build_model_result(
                path_b_result, fe.scaler_means, fe.scaler_stds, fe.categorical_map
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _aggregate_monthly_summary(lift_results: list[dict]) -> list[MonthlySummary]:
    """Aggregate lift results into monthly summaries."""
    by_month: dict[str, dict] = defaultdict(
        lambda: {
            "predictedAction": 0.0,
            "counterfactualAction": 0.0,
            "incrementalAction": 0.0,
            "predictedOutcome": 0.0,
            "counterfactualOutcome": 0.0,
            "incrementalOutcome": 0.0,
            "observedAction": 0.0,
            "observedOutcome": 0.0,
        }
    )

    for r in lift_results:
        month_key = r["month"][:7]  # YYYY-MM
        agg = by_month[month_key]
        agg["predictedAction"] += r["predictedAction"]
        agg["counterfactualAction"] += r["counterfactualAction"]
        agg["incrementalAction"] += r["incrementalAction"]
        agg["predictedOutcome"] += r["predictedOutcome"]
        agg["counterfactualOutcome"] += r["counterfactualOutcome"]
        agg["incrementalOutcome"] += r["incrementalOutcome"]
        agg["observedAction"] += r["observedAction"]
        agg["observedOutcome"] += r["observedOutcome"]

    summaries = []
    for month in sorted(by_month.keys()):
        agg = by_month[month]
        lift_pct_action = (
            (agg["incrementalAction"] / abs(agg["counterfactualAction"])) * 100
            if abs(agg["counterfactualAction"]) > 0
            else 0.0
        )
        lift_pct_outcome = (
            (agg["incrementalOutcome"] / abs(agg["counterfactualOutcome"])) * 100
            if abs(agg["counterfactualOutcome"]) > 0
            else 0.0
        )

        summaries.append(
            MonthlySummary(
                month=month,
                totalPredictedAction=agg["predictedAction"],
                totalCounterfactualAction=agg["counterfactualAction"],
                totalIncrementalAction=agg["incrementalAction"],
                totalPredictedOutcome=agg["predictedOutcome"],
                totalCounterfactualOutcome=agg["counterfactualOutcome"],
                totalIncrementalOutcome=agg["incrementalOutcome"],
                totalObservedAction=agg["observedAction"],
                totalObservedOutcome=agg["observedOutcome"],
                liftPercentAction=lift_pct_action,
                liftPercentOutcome=lift_pct_outcome,
            )
        )

    return summaries
