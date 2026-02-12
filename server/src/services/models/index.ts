/**
 * Model Layer — Path A and Path B Regression Models
 *
 * Two-path causal framework:
 *
 *   Path A (Suggestion → Action):
 *     Trains a regression model to predict action_count from features
 *     that include suggestion_count, percent_other_hcps_suggested, and
 *     other HCP-level covariates.
 *
 *   Path B (Action → Outcome):
 *     Trains a regression model to predict outcome_count (TRX or NBRX)
 *     from features that include the predicted/counterfactual action_count
 *     and other covariates.
 *
 * Both models use OLS linear regression with ridge regularization,
 * implemented via the normal equation: β = (XᵀX + λI)⁻¹ Xᵀy
 */

import { solveOLS, predict } from '../../utils/linearAlgebra';
import {
  EngineeredRow,
  TrainedModel,
  FeatureImportanceEntry,
} from '../../utils/types';

/**
 * PathAActionModel — predicts action_count from suggestion features.
 *
 * Target: action_count
 * Features: all engineered features (includes suggestion_count,
 *           percent_other_hcps_suggested, tenure, priors, etc.)
 */
export class PathAActionModel {
  private coefficients: number[] = [];
  private intercept: number = 0;
  private featureNames: string[] = [];

  /**
   * Train the Path A model using OLS regression.
   *
   * Prepends an intercept column (column of 1s) to the design matrix,
   * solves for β, and extracts feature importance from |β| values.
   */
  train(rows: EngineeredRow[]): TrainedModel {
    this.featureNames = rows[0].featureNames;
    const n = rows.length;
    const p = this.featureNames.length;

    // Build design matrix X with intercept column
    const X: number[][] = rows.map((row) => [1, ...row.features]);
    const y: number[] = rows.map((row) => row.actionCount);

    // Solve OLS: β = (XᵀX + λI)⁻¹ Xᵀy
    const beta = solveOLS(X, y);
    this.intercept = beta[0];
    this.coefficients = beta.slice(1);

    // Compute predictions and metrics
    const predictions = predict(X, beta);
    const { rSquared, mae } = this.computeMetrics(y, predictions);

    // Extract feature importance from absolute coefficient values
    const importance = this.extractFeatureImportance();

    return {
      modelName: 'PathA_Action',
      coefficients: this.coefficients,
      intercept: this.intercept,
      featureNames: this.featureNames,
      scalerState: { means: {}, stds: {} }, // Filled by caller
      categoricalMap: {},
      featureImportance: importance,
      rSquared,
      mae,
    };
  }

  /**
   * Generate predictions for given feature rows.
   */
  predict(rows: EngineeredRow[]): number[] {
    const X = rows.map((row) => [1, ...row.features]);
    const beta = [this.intercept, ...this.coefficients];
    return predict(X, beta);
  }

  /**
   * Generate predictions from raw feature vectors (for counterfactual).
   */
  predictFromFeatures(featureVectors: number[][]): number[] {
    const X = featureVectors.map((f) => [1, ...f]);
    const beta = [this.intercept, ...this.coefficients];
    return predict(X, beta);
  }

  /**
   * Compute R² and MAE between actual and predicted values.
   */
  private computeMetrics(
    actual: number[],
    predicted: number[]
  ): { rSquared: number; mae: number } {
    const n = actual.length;
    const mean = actual.reduce((s, v) => s + v, 0) / n;

    let ssRes = 0;
    let ssTot = 0;
    let absError = 0;

    for (let i = 0; i < n; i++) {
      ssRes += (actual[i] - predicted[i]) ** 2;
      ssTot += (actual[i] - mean) ** 2;
      absError += Math.abs(actual[i] - predicted[i]);
    }

    const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    const mae = absError / n;

    return { rSquared, mae };
  }

  /**
   * Extract feature importance ranked by |coefficient|.
   * Larger absolute coefficients indicate greater influence
   * on the predicted action_count.
   */
  private extractFeatureImportance(): FeatureImportanceEntry[] {
    const entries = this.featureNames.map((name, i) => ({
      featureName: name,
      importance: Math.abs(this.coefficients[i] || 0),
      rank: 0,
    }));

    // Sort descending by importance and assign ranks
    entries.sort((a, b) => b.importance - a.importance);
    entries.forEach((entry, idx) => {
      entry.rank = idx + 1;
    });

    return entries;
  }

  getCoefficients(): number[] {
    return this.coefficients;
  }

  getIntercept(): number {
    return this.intercept;
  }

  getFeatureNames(): string[] {
    return this.featureNames;
  }
}

/**
 * PathBOutcomeModel — predicts outcome_count (TRX or NBRX) from
 * action and other features.
 *
 * Target: outcome_count
 * Features: same engineered features, but during counterfactual
 *           simulation the action_count feature is replaced with
 *           predicted or counterfactual action values.
 */
export class PathBOutcomeModel {
  private coefficients: number[] = [];
  private intercept: number = 0;
  private featureNames: string[] = [];
  private outcomeType: 'trx' | 'nbrx';

  constructor(outcomeType: 'trx' | 'nbrx') {
    this.outcomeType = outcomeType;
  }

  /**
   * Train the Path B model using OLS regression.
   *
   * Similar to Path A, but targets outcome_count instead of action_count.
   * The action_count feature in the input already contains the observed
   * action values; during inference, it will be substituted with
   * predicted or counterfactual values.
   */
  train(rows: EngineeredRow[]): TrainedModel {
    this.featureNames = rows[0].featureNames;

    // Build design matrix X with intercept column
    const X: number[][] = rows.map((row) => [1, ...row.features]);
    const y: number[] = rows.map((row) => row.outcomeCount);

    // Solve OLS
    const beta = solveOLS(X, y);
    this.intercept = beta[0];
    this.coefficients = beta.slice(1);

    // Compute predictions and metrics
    const predictions = predict(X, beta);
    const { rSquared, mae } = this.computeMetrics(y, predictions);

    const importance = this.extractFeatureImportance();

    const modelName =
      this.outcomeType === 'trx' ? 'PathB_TRX' : 'PathB_NBRX';

    return {
      modelName,
      coefficients: this.coefficients,
      intercept: this.intercept,
      featureNames: this.featureNames,
      scalerState: { means: {}, stds: {} },
      categoricalMap: {},
      featureImportance: importance,
      rSquared,
      mae,
    };
  }

  /**
   * Generate predictions for given feature rows.
   */
  predict(rows: EngineeredRow[]): number[] {
    const X = rows.map((row) => [1, ...row.features]);
    const beta = [this.intercept, ...this.coefficients];
    return predict(X, beta);
  }

  /**
   * Generate predictions from raw feature vectors.
   * Used during counterfactual simulation where the action
   * feature has been substituted.
   */
  predictFromFeatures(featureVectors: number[][]): number[] {
    const X = featureVectors.map((f) => [1, ...f]);
    const beta = [this.intercept, ...this.coefficients];
    return predict(X, beta);
  }

  private computeMetrics(
    actual: number[],
    predicted: number[]
  ): { rSquared: number; mae: number } {
    const n = actual.length;
    const mean = actual.reduce((s, v) => s + v, 0) / n;

    let ssRes = 0;
    let ssTot = 0;
    let absError = 0;

    for (let i = 0; i < n; i++) {
      ssRes += (actual[i] - predicted[i]) ** 2;
      ssTot += (actual[i] - mean) ** 2;
      absError += Math.abs(actual[i] - predicted[i]);
    }

    const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    const mae = absError / n;

    return { rSquared, mae };
  }

  private extractFeatureImportance(): FeatureImportanceEntry[] {
    const entries = this.featureNames.map((name, i) => ({
      featureName: name,
      importance: Math.abs(this.coefficients[i] || 0),
      rank: 0,
    }));

    entries.sort((a, b) => b.importance - a.importance);
    entries.forEach((entry, idx) => {
      entry.rank = idx + 1;
    });

    return entries;
  }

  getCoefficients(): number[] {
    return this.coefficients;
  }

  getIntercept(): number {
    return this.intercept;
  }

  getFeatureNames(): string[] {
    return this.featureNames;
  }
}
