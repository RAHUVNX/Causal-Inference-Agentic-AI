/**
 * Counterfactual Simulation Engine
 *
 * Implements the core causal inference logic:
 *
 * 1. Zero out suggestion features → generate counterfactual_action_count
 *    (What would the action have been WITHOUT the suggestion?)
 *
 * 2. Replace the action feature in the outcome model with
 *    predicted vs counterfactual action → generate outcome predictions
 *    (What would the outcome have been with/without the suggested action?)
 *
 * The incremental lift is the difference:
 *   incremental_action  = predicted_action  − counterfactual_action
 *   incremental_outcome = predicted_outcome − counterfactual_outcome
 */

import {
  SUGGESTION_FEATURE_NAME,
  PERCENT_OTHER_HCP_FEATURE_NAME,
} from '../featureEngineer';
import { PathAActionModel, PathBOutcomeModel } from '../models';
import { EngineeredRow, ScalerState, LiftResultRow } from '../../utils/types';

/**
 * Names of features related to suggestions that should be zeroed
 * when constructing the counterfactual world.
 */
const SUGGESTION_RELATED_FEATURES = [
  SUGGESTION_FEATURE_NAME,
  PERCENT_OTHER_HCP_FEATURE_NAME,
  'totalSuggestions',
];

export class CounterfactualEngine {
  private pathAModel: PathAActionModel;
  private pathBModel: PathBOutcomeModel;
  private scalerState: ScalerState;

  constructor(
    pathAModel: PathAActionModel,
    pathBModel: PathBOutcomeModel,
    scalerState: ScalerState
  ) {
    this.pathAModel = pathAModel;
    this.pathBModel = pathBModel;
    this.scalerState = scalerState;
  }

  /**
   * Run the full counterfactual simulation.
   *
   * For each observation:
   *
   *   PATH A — Action prediction:
   *     predicted_action       = PathA.predict(original_features)
   *     counterfactual_action  = PathA.predict(zeroed_suggestion_features)
   *     incremental_action     = predicted_action − counterfactual_action
   *
   *   PATH B — Outcome prediction:
   *     predicted_outcome      = PathB.predict(features_with_predicted_action)
   *     counterfactual_outcome = PathB.predict(features_with_counterfactual_action)
   *     incremental_outcome    = predicted_outcome − counterfactual_outcome
   */
  simulate(
    rows: EngineeredRow[],
    config: { suggestionType: string; actionType: string; outcomeType: string }
  ): LiftResultRow[] {
    const featureNames = rows[0].featureNames;

    // ── Step 1: Path A — Predict action with original features ──────
    const predictedActions = this.pathAModel.predict(rows);

    // ── Step 2: Zero out suggestion features for counterfactual ─────
    const counterfactualFeatures = this.zeroOutSuggestionFeatures(
      rows,
      featureNames
    );

    // ── Step 3: Path A — Predict counterfactual action ──────────────
    const counterfactualActions =
      this.pathAModel.predictFromFeatures(counterfactualFeatures);

    // ── Step 4: Path B — Predict outcome with predicted action ──────
    //    Replace the action_count feature with predicted_action
    const featuresWithPredictedAction = this.substituteActionFeature(
      rows,
      featureNames,
      predictedActions
    );
    const predictedOutcomes =
      this.pathBModel.predictFromFeatures(featuresWithPredictedAction);

    // ── Step 5: Path B — Predict outcome with counterfactual action ─
    //    Replace the action_count feature with counterfactual_action
    const featuresWithCounterfactualAction = this.substituteActionFeature(
      rows,
      featureNames,
      counterfactualActions
    );
    const counterfactualOutcomes =
      this.pathBModel.predictFromFeatures(featuresWithCounterfactualAction);

    // ── Step 6: Compute incremental lift for each row ───────────────
    const results: LiftResultRow[] = rows.map((row, i) => {
      const incrementalAction = predictedActions[i] - counterfactualActions[i];
      const incrementalOutcome =
        predictedOutcomes[i] - counterfactualOutcomes[i];

      return {
        hcpId: row.hcpId,
        month: row.month,
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        theme: row.theme,
        predictedAction: predictedActions[i],
        counterfactualAction: counterfactualActions[i],
        incrementalAction,
        predictedOutcome: predictedOutcomes[i],
        counterfactualOutcome: counterfactualOutcomes[i],
        incrementalOutcome,
        observedAction: row.observedAction,
        observedOutcome: row.observedOutcome,
      };
    });

    return results;
  }

  /**
   * Zero out all suggestion-related features in the feature vectors.
   *
   * This creates the "counterfactual world" where no suggestions were made.
   * Suggestion features are set to their standardized zero value:
   *   standardized_zero = (0 − mean) / std
   *
   * This accounts for the fact that features have been z-score standardized.
   */
  private zeroOutSuggestionFeatures(
    rows: EngineeredRow[],
    featureNames: string[]
  ): number[][] {
    // Find indices of suggestion-related features
    const suggestionIndices: number[] = [];
    for (let i = 0; i < featureNames.length; i++) {
      if (SUGGESTION_RELATED_FEATURES.includes(featureNames[i])) {
        suggestionIndices.push(i);
      }
    }

    return rows.map((row) => {
      const features = [...row.features];
      for (const idx of suggestionIndices) {
        const name = featureNames[idx];
        // Standardized value of zero: (0 - mean) / std
        const mean = this.scalerState.means[name] || 0;
        const std = this.scalerState.stds[name] || 1;
        features[idx] = (0 - mean) / std;
      }
      return features;
    });
  }

  /**
   * Replace the actionCount feature in each row's feature vector
   * with a given substitute value (predicted or counterfactual action).
   *
   * The substitute values are standardized using the scaler state
   * for the actionCount feature before insertion.
   */
  private substituteActionFeature(
    rows: EngineeredRow[],
    featureNames: string[],
    substituteActions: number[]
  ): number[][] {
    const actionIdx = featureNames.indexOf('actionCount');
    if (actionIdx === -1) {
      // If actionCount is not a separate feature, return original features
      return rows.map((row) => [...row.features]);
    }

    const mean = this.scalerState.means['actionCount'] || 0;
    const std = this.scalerState.stds['actionCount'] || 1;

    return rows.map((row, i) => {
      const features = [...row.features];
      // Standardize the substitute action value
      features[actionIdx] = (substituteActions[i] - mean) / std;
      return features;
    });
  }
}
