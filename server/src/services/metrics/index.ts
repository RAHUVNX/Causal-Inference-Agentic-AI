/**
 * Metrics Layer
 *
 * Computes evaluation metrics for trained models:
 *   - R² (coefficient of determination) for regression quality
 *   - MAE (mean absolute error)
 *   - RMSE (root mean squared error)
 *   - Accuracy, Precision, Recall, F1 (for binarized lift classification)
 *   - Feature importance ranking
 *
 * The classification metrics treat incremental lift as a binary signal:
 *   positive lift (incremental > 0) → class 1
 *   zero or negative lift           → class 0
 *
 * This allows reporting both regression and classification perspectives
 * on the causal lift estimates.
 */

import { LiftResultRow, FeatureImportanceEntry, TrainedModel } from '../../utils/types';

export interface ModelMetrics {
  rSquared: number;
  mae: number;
  rmse: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface FullMetricsReport {
  pathA: ModelMetrics;
  pathB: ModelMetrics;
  featureImportancePathA: FeatureImportanceEntry[];
  featureImportancePathB: FeatureImportanceEntry[];
}

/**
 * Compute R² (coefficient of determination).
 *
 * R² = 1 − (SS_res / SS_tot)
 * where SS_res = Σ(actual − predicted)² and SS_tot = Σ(actual − mean)²
 */
export function computeRSquared(actual: number[], predicted: number[]): number {
  const n = actual.length;
  if (n === 0) return 0;

  const mean = actual.reduce((s, v) => s + v, 0) / n;
  let ssRes = 0;
  let ssTot = 0;

  for (let i = 0; i < n; i++) {
    ssRes += (actual[i] - predicted[i]) ** 2;
    ssTot += (actual[i] - mean) ** 2;
  }

  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

/**
 * Compute MAE (mean absolute error).
 */
export function computeMAE(actual: number[], predicted: number[]): number {
  const n = actual.length;
  if (n === 0) return 0;
  return actual.reduce((s, v, i) => s + Math.abs(v - predicted[i]), 0) / n;
}

/**
 * Compute RMSE (root mean squared error).
 */
export function computeRMSE(actual: number[], predicted: number[]): number {
  const n = actual.length;
  if (n === 0) return 0;
  const mse = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0) / n;
  return Math.sqrt(mse);
}

/**
 * Compute binary classification metrics by thresholding incremental lift.
 *
 * Binarization:
 *   actual_positive  = observed > 0
 *   predicted_positive = incremental > 0
 */
export function computeClassificationMetrics(
  observed: number[],
  incremental: number[]
): { accuracy: number; precision: number; recall: number; f1: number } {
  const n = observed.length;
  if (n === 0) return { accuracy: 0, precision: 0, recall: 0, f1: 0 };

  let tp = 0; // True positive
  let fp = 0; // False positive
  let fn = 0; // False negative
  let tn = 0; // True negative

  for (let i = 0; i < n; i++) {
    const actualPositive = observed[i] > 0;
    const predictedPositive = incremental[i] > 0;

    if (predictedPositive && actualPositive) tp++;
    else if (predictedPositive && !actualPositive) fp++;
    else if (!predictedPositive && actualPositive) fn++;
    else tn++;
  }

  const accuracy = (tp + tn) / n;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 =
    precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : 0;

  return { accuracy, precision, recall, f1 };
}

/**
 * Compute full metrics report for both Path A and Path B models.
 */
export function computeFullMetrics(
  liftResults: LiftResultRow[],
  pathAModel: TrainedModel,
  pathBModel: TrainedModel
): FullMetricsReport {
  const observedActions = liftResults.map((r) => r.observedAction);
  const predictedActions = liftResults.map((r) => r.predictedAction);
  const incrementalActions = liftResults.map((r) => r.incrementalAction);

  const observedOutcomes = liftResults.map((r) => r.observedOutcome);
  const predictedOutcomes = liftResults.map((r) => r.predictedOutcome);
  const incrementalOutcomes = liftResults.map((r) => r.incrementalOutcome);

  // Path A metrics
  const pathARSquared = computeRSquared(observedActions, predictedActions);
  const pathAMAE = computeMAE(observedActions, predictedActions);
  const pathARMSE = computeRMSE(observedActions, predictedActions);
  const pathAClassification = computeClassificationMetrics(
    observedActions,
    incrementalActions
  );

  // Path B metrics
  const pathBRSquared = computeRSquared(observedOutcomes, predictedOutcomes);
  const pathBMAE = computeMAE(observedOutcomes, predictedOutcomes);
  const pathBRMSE = computeRMSE(observedOutcomes, predictedOutcomes);
  const pathBClassification = computeClassificationMetrics(
    observedOutcomes,
    incrementalOutcomes
  );

  return {
    pathA: {
      rSquared: pathARSquared,
      mae: pathAMAE,
      rmse: pathARMSE,
      ...pathAClassification,
    },
    pathB: {
      rSquared: pathBRSquared,
      mae: pathBMAE,
      rmse: pathBRMSE,
      ...pathBClassification,
    },
    featureImportancePathA: pathAModel.featureImportance,
    featureImportancePathB: pathBModel.featureImportance,
  };
}
