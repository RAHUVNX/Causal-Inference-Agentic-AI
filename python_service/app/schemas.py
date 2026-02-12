"""
Pydantic schemas for the Python ML service API.
"""

from pydantic import BaseModel
from typing import Optional


class ObservationRow(BaseModel):
    hcpId: str
    month: str
    suggestionType: str
    actionType: str
    outcomeType: str
    suggestionCount: float
    actionCount: float
    outcomeCount: float
    specialtyCode: Optional[str] = None
    regionCode: Optional[str] = None
    tenureMonths: Optional[float] = None
    priorTrx: Optional[float] = None
    priorNbrx: Optional[float] = None
    totalSuggestions: Optional[float] = None
    theme: Optional[str] = None


class FilterTriangle(BaseModel):
    suggestionType: str
    actionType: str
    outcomeType: str


class TrainRequest(BaseModel):
    observations: list[ObservationRow]
    filter: FilterTriangle
    adstockEnabled: bool = False
    adstockDecay: float = 0.5
    theme: Optional[str] = None


class LiftRequest(BaseModel):
    observations: list[ObservationRow]
    filter: FilterTriangle
    adstockEnabled: bool = False
    adstockDecay: float = 0.5
    theme: Optional[str] = None


class FeatureImportanceEntry(BaseModel):
    featureName: str
    importance: float
    rank: int


class ModelMetrics(BaseModel):
    rSquared: float
    mae: float
    rmse: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class TrainedModelResult(BaseModel):
    modelName: str
    coefficients: list[float]
    intercept: float
    featureNames: list[str]
    scalerMeans: dict[str, float]
    scalerStds: dict[str, float]
    categoricalMap: dict[str, dict[str, list[int]]]
    featureImportance: list[FeatureImportanceEntry]
    rSquared: float
    mae: float


class TrainResponse(BaseModel):
    pathA: TrainedModelResult
    pathB: TrainedModelResult


class LiftResultRow(BaseModel):
    hcpId: str
    month: str
    suggestionType: str
    actionType: str
    outcomeType: str
    theme: Optional[str] = None
    predictedAction: float
    counterfactualAction: float
    incrementalAction: float
    predictedOutcome: float
    counterfactualOutcome: float
    incrementalOutcome: float
    observedAction: float
    observedOutcome: float


class MonthlySummary(BaseModel):
    month: str
    totalPredictedAction: float
    totalCounterfactualAction: float
    totalIncrementalAction: float
    totalPredictedOutcome: float
    totalCounterfactualOutcome: float
    totalIncrementalOutcome: float
    totalObservedAction: float
    totalObservedOutcome: float
    liftPercentAction: float
    liftPercentOutcome: float


class LiftResponse(BaseModel):
    liftResults: list[LiftResultRow]
    monthlySummary: list[MonthlySummary]
    metrics: dict[str, ModelMetrics]
    pathA: TrainedModelResult
    pathB: TrainedModelResult
