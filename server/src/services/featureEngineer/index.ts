/**
 * Feature Engineering Pipeline
 *
 * Mirrors the Spark FeatureEngineer component. Responsible for:
 *  1. Filter triangle: filter by suggestion_type, action_type, outcome_type
 *  2. Study period filtering
 *  3. Feature selection
 *  4. percent_other_hcps_suggested calculation
 *  5. Optional adstock transformation
 *  6. Categorical encoding (one-hot)
 *  7. Standardization (z-score scaling)
 */

import { Observation } from '@prisma/client';
import {
  EngineeredRow,
  FeatureEngineerConfig,
  ScalerState,
  CategoricalMap,
} from '../../utils/types';

// Numeric columns eligible for features
const NUMERIC_FEATURE_COLS = [
  'suggestionCount',
  'actionCount',
  'tenureMonths',
  'priorTrx',
  'priorNbrx',
  'totalSuggestions',
  'percentOtherHcpsSuggested',
] as const;

// Categorical columns to one-hot encode
const CATEGORICAL_COLS = ['specialtyCode', 'regionCode'] as const;

// The suggestion feature that gets zeroed in counterfactual
export const SUGGESTION_FEATURE_NAME = 'suggestionCount';
export const PERCENT_OTHER_HCP_FEATURE_NAME = 'percentOtherHcpsSuggested';

export class FeatureEngineer {
  private config: FeatureEngineerConfig;
  private scalerState: ScalerState | null = null;
  private categoricalMap: CategoricalMap | null = null;

  constructor(config: FeatureEngineerConfig) {
    this.config = config;
  }

  /**
   * Run the full feature engineering pipeline on raw observations.
   *
   * Steps:
   *  1. Apply filter triangle (suggestion_type, action_type, outcome_type)
   *  2. Apply study period filter
   *  3. Apply optional theme filter
   *  4. Compute percent_other_hcps_suggested
   *  5. Apply optional adstock transformation to suggestion_count
   *  6. Build categorical encoding map
   *  7. Standardize numeric features
   *  8. Assemble final feature vectors
   */
  engineer(observations: Observation[]): {
    rows: EngineeredRow[];
    scalerState: ScalerState;
    categoricalMap: CategoricalMap;
    featureNames: string[];
  } {
    // Step 1: Filter triangle
    let filtered = this.applyFilterTriangle(observations);

    // Step 2: Study period filtering
    filtered = this.applyStudyPeriod(filtered);

    // Step 3: Theme filter
    if (this.config.theme) {
      filtered = filtered.filter(
        (obs) => obs.theme?.toLowerCase() === this.config.theme!.toLowerCase()
      );
    }

    if (filtered.length === 0) {
      throw new Error(
        'No observations remain after filtering. Check filter triangle and study period.'
      );
    }

    // Step 4: Compute percent_other_hcps_suggested per month
    const withPercentOther = this.computePercentOtherHcpsSuggested(filtered);

    // Step 5: Optional adstock transformation
    const withAdstock = this.config.adstockEnabled
      ? this.applyAdstock(withPercentOther, this.config.adstockDecay)
      : withPercentOther;

    // Step 6: Build categorical encoding map
    this.categoricalMap = this.buildCategoricalMap(withAdstock);

    // Step 7 & 8: Standardize and assemble feature vectors
    const { rows, featureNames, scalerState } =
      this.standardizeAndAssemble(withAdstock);

    this.scalerState = scalerState;

    return { rows, scalerState, categoricalMap: this.categoricalMap, featureNames };
  }

  /**
   * Filter triangle: keep only rows matching the configured
   * suggestion_type, action_type, and outcome_type.
   */
  private applyFilterTriangle(observations: Observation[]): Observation[] {
    return observations.filter(
      (obs) =>
        obs.suggestionType === this.config.filter.suggestionType &&
        obs.actionType === this.config.filter.actionType &&
        obs.outcomeType === this.config.filter.outcomeType
    );
  }

  /**
   * Study period filtering: keep observations within the
   * configured start/end date range.
   */
  private applyStudyPeriod(observations: Observation[]): Observation[] {
    let result = observations;
    if (this.config.studyPeriodStart) {
      const start = new Date(this.config.studyPeriodStart);
      result = result.filter((obs) => obs.month >= start);
    }
    if (this.config.studyPeriodEnd) {
      const end = new Date(this.config.studyPeriodEnd);
      result = result.filter((obs) => obs.month <= end);
    }
    return result;
  }

  /**
   * Compute percent_other_hcps_suggested:
   *   For each HCP in a given month, calculate what percentage of
   *   other HCPs also received a suggestion.
   *
   *   percent_other = (count_of_other_suggested_hcps) / (total_hcps - 1)
   *
   * This captures network/peer effects in the causal model.
   */
  private computePercentOtherHcpsSuggested(
    observations: Observation[]
  ): (Observation & { percentOtherHcpsSuggested: number })[] {
    // Group by month
    const byMonth = new Map<string, Observation[]>();
    for (const obs of observations) {
      const key = obs.month.toISOString().slice(0, 7); // YYYY-MM
      if (!byMonth.has(key)) byMonth.set(key, []);
      byMonth.get(key)!.push(obs);
    }

    return observations.map((obs) => {
      const monthKey = obs.month.toISOString().slice(0, 7);
      const monthObs = byMonth.get(monthKey) || [];
      const totalHcps = new Set(monthObs.map((o) => o.hcpId)).size;
      const suggestedHcps = new Set(
        monthObs.filter((o) => o.suggestionCount > 0).map((o) => o.hcpId)
      ).size;

      // Exclude the current HCP from the "other" count
      const isSuggested = obs.suggestionCount > 0 ? 1 : 0;
      const otherSuggested = suggestedHcps - isSuggested;
      const otherTotal = totalHcps - 1;

      const percentOtherHcpsSuggested =
        otherTotal > 0 ? otherSuggested / otherTotal : 0;

      return { ...obs, percentOtherHcpsSuggested };
    });
  }

  /**
   * Adstock transformation:
   *   Applies geometric decay to suggestion_count over time per HCP.
   *   adstocked_t = suggestion_t + λ * adstocked_{t-1}
   *
   * This models the decaying influence of past suggestions.
   */
  private applyAdstock<T extends Observation & { percentOtherHcpsSuggested: number }>(
    observations: T[],
    decay: number
  ): T[] {
    // Group by HCP and sort by month
    const byHcp = new Map<string, T[]>();
    for (const obs of observations) {
      if (!byHcp.has(obs.hcpId)) byHcp.set(obs.hcpId, []);
      byHcp.get(obs.hcpId)!.push(obs);
    }

    const result: T[] = [];
    for (const [, hcpObs] of byHcp) {
      // Sort by month ascending
      hcpObs.sort((a, b) => a.month.getTime() - b.month.getTime());

      let prevAdstock = 0;
      for (const obs of hcpObs) {
        const adstocked = obs.suggestionCount + decay * prevAdstock;
        result.push({
          ...obs,
          suggestionCount: adstocked,
        });
        prevAdstock = adstocked;
      }
    }

    return result;
  }

  /**
   * Build one-hot encoding map for categorical columns.
   * Maps each unique category to a binary vector.
   */
  private buildCategoricalMap(
    observations: (Observation & { percentOtherHcpsSuggested: number })[]
  ): CategoricalMap {
    const catMap: CategoricalMap = {};

    for (const col of CATEGORICAL_COLS) {
      const uniqueValues = [
        ...new Set(
          observations
            .map((obs) => (obs as Record<string, unknown>)[col] as string | null | undefined)
            .filter((v): v is string => v != null && v !== '')
        ),
      ].sort();

      catMap[col] = {};
      for (let i = 0; i < uniqueValues.length; i++) {
        const oneHot = new Array(uniqueValues.length).fill(0);
        oneHot[i] = 1;
        catMap[col][uniqueValues[i]] = oneHot;
      }
    }

    return catMap;
  }

  /**
   * Standardize numeric features (z-score) and assemble
   * the final feature vectors combining numeric + one-hot features.
   */
  private standardizeAndAssemble(
    observations: (Observation & { percentOtherHcpsSuggested: number })[]
  ): {
    rows: EngineeredRow[];
    featureNames: string[];
    scalerState: ScalerState;
  } {
    // Collect numeric values for standardization
    const numericData: Record<string, number[]> = {};
    for (const col of NUMERIC_FEATURE_COLS) {
      numericData[col] = observations.map((obs) => {
        const val = (obs as Record<string, unknown>)[col];
        return typeof val === 'number' ? val : 0;
      });
    }

    // Compute mean and std for each numeric column
    const means: Record<string, number> = {};
    const stds: Record<string, number> = {};
    for (const col of NUMERIC_FEATURE_COLS) {
      const values = numericData[col];
      const mean = values.reduce((s, v) => s + v, 0) / values.length;
      const variance =
        values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
      const std = Math.sqrt(variance) || 1; // Avoid division by zero
      means[col] = mean;
      stds[col] = std;
    }

    const scalerState: ScalerState = { means, stds };

    // Build feature names: numeric (standardized) + categorical (one-hot)
    const featureNames: string[] = [...NUMERIC_FEATURE_COLS];
    for (const col of CATEGORICAL_COLS) {
      const categories = Object.keys(this.categoricalMap![col] || {});
      for (const cat of categories) {
        featureNames.push(`${col}_${cat}`);
      }
    }

    // Assemble rows
    const rows: EngineeredRow[] = observations.map((obs, idx) => {
      const features: number[] = [];

      // Standardized numeric features
      for (const col of NUMERIC_FEATURE_COLS) {
        const raw = numericData[col][idx];
        features.push((raw - means[col]) / stds[col]);
      }

      // One-hot encoded categorical features
      for (const col of CATEGORICAL_COLS) {
        const val = (obs as Record<string, unknown>)[col] as string | null | undefined;
        const categories = Object.keys(this.categoricalMap![col] || {});
        if (val && this.categoricalMap![col][val]) {
          features.push(...this.categoricalMap![col][val]);
        } else {
          // Unknown category → all zeros
          features.push(...new Array(categories.length).fill(0));
        }
      }

      return {
        hcpId: obs.hcpId,
        month: obs.month.toISOString(),
        features,
        featureNames,
        actionCount: obs.actionCount,
        outcomeCount: obs.outcomeCount,
        suggestionCount: obs.suggestionCount,
        observedAction: obs.actionCount,
        observedOutcome: obs.outcomeCount,
        theme: obs.theme || undefined,
      };
    });

    return { rows, featureNames, scalerState };
  }

  getScalerState(): ScalerState | null {
    return this.scalerState;
  }

  getCategoricalMap(): CategoricalMap | null {
    return this.categoricalMap;
  }
}
