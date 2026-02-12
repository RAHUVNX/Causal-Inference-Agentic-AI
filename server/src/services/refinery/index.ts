/**
 * Refinery — Orchestration Layer
 *
 * Coordinates the full causal lift pipeline:
 *
 *   1. Feature Engineering  → prepare data for modeling
 *   2. Train Path A Model   → Suggestion → Action
 *   3. Train Path B Model   → Action → Outcome (TRX/NBRX)
 *   4. Counterfactual Sim   → zero suggestions, compute deltas
 *   5. Compute Metrics      → evaluate model quality
 *   6. Aggregate Results    → monthly summaries with lift %
 *
 * This mirrors the Spark Refinery component that orchestrates
 * the end-to-end causal inference pipeline.
 */

import { Observation } from '@prisma/client';
import { FeatureEngineer } from '../featureEngineer';
import { PathAActionModel, PathBOutcomeModel } from '../models';
import { CounterfactualEngine } from '../counterfactual';
import { computeFullMetrics, FullMetricsReport } from '../metrics';
import {
  RunConfig,
  PipelineResult,
  LiftResultRow,
  MonthlySummary,
  TrainedModel,
  FeatureEngineerConfig,
} from '../../utils/types';

/**
 * Generate a unique run ID for tracking pipeline executions.
 */
function generateRunId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `run_${timestamp}_${random}`;
}

export class Refinery {
  /**
   * Execute the complete causal lift pipeline.
   *
   * Orchestration flow:
   *   observations → FeatureEngineer → PathAModel.train → PathBModel.train
   *     → CounterfactualEngine.simulate → metrics + aggregation
   *
   * @param observations  Raw observation data from the database
   * @param config        Run configuration (filter triangle, adstock, etc.)
   * @returns             Full pipeline result including models, lift, and summaries
   */
  run(
    observations: Observation[],
    config: RunConfig
  ): {
    result: PipelineResult;
    metrics: FullMetricsReport;
  } {
    const runId = generateRunId();

    // ─── Step 1: Feature Engineering ────────────────────────────
    // Apply filter triangle, compute percent_other_hcps_suggested,
    // optional adstock, categorical encoding, and standardization.
    const feConfig: FeatureEngineerConfig = {
      filter: {
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
      },
      adstockEnabled: config.adstockEnabled || false,
      adstockDecay: config.adstockDecay || 0.5,
      theme: config.theme,
    };

    const featureEngineer = new FeatureEngineer(feConfig);
    const { rows, scalerState, categoricalMap, featureNames } =
      featureEngineer.engineer(observations);

    // ─── Step 2: Train Path A Model (Suggestion → Action) ──────
    // Predicts action_count from all features including suggestion features.
    const pathAModel = new PathAActionModel();
    const pathAResult: TrainedModel = pathAModel.train(rows);
    pathAResult.scalerState = scalerState;
    pathAResult.categoricalMap = categoricalMap;

    // ─── Step 3: Train Path B Model (Action → Outcome) ─────────
    // Predicts outcome_count (TRX/NBRX) from all features.
    const pathBModel = new PathBOutcomeModel(config.outcomeType);
    const pathBResult: TrainedModel = pathBModel.train(rows);
    pathBResult.scalerState = scalerState;
    pathBResult.categoricalMap = categoricalMap;

    // ─── Step 4: Counterfactual Simulation ─────────────────────
    // Zero out suggestion features → counterfactual_action
    // Substitute predicted/counterfactual action into outcome model
    // → predicted_outcome and counterfactual_outcome
    // Compute incremental deltas
    const counterfactualEngine = new CounterfactualEngine(
      pathAModel,
      pathBModel,
      scalerState
    );

    const liftResults: LiftResultRow[] = counterfactualEngine.simulate(rows, {
      suggestionType: config.suggestionType,
      actionType: config.actionType,
      outcomeType: config.outcomeType,
    });

    // ─── Step 5: Compute Metrics ───────────────────────────────
    const metrics = computeFullMetrics(liftResults, pathAResult, pathBResult);

    // ─── Step 6: Aggregate Monthly Summaries ───────────────────
    const monthlySummary = this.aggregateMonthlySummary(liftResults);

    const result: PipelineResult = {
      runId,
      pathAModel: pathAResult,
      pathBModel: pathBResult,
      liftResults,
      monthlySummary,
    };

    return { result, metrics };
  }

  /**
   * Aggregate lift results into monthly summaries.
   *
   * For each month, computes:
   *   - Total predicted, counterfactual, and incremental values
   *   - Lift percentage: (incremental / counterfactual) × 100
   */
  private aggregateMonthlySummary(results: LiftResultRow[]): MonthlySummary[] {
    const byMonth = new Map<
      string,
      {
        predictedAction: number;
        counterfactualAction: number;
        incrementalAction: number;
        predictedOutcome: number;
        counterfactualOutcome: number;
        incrementalOutcome: number;
        observedAction: number;
        observedOutcome: number;
      }
    >();

    for (const r of results) {
      // Group by YYYY-MM
      const monthKey = r.month.slice(0, 7);

      if (!byMonth.has(monthKey)) {
        byMonth.set(monthKey, {
          predictedAction: 0,
          counterfactualAction: 0,
          incrementalAction: 0,
          predictedOutcome: 0,
          counterfactualOutcome: 0,
          incrementalOutcome: 0,
          observedAction: 0,
          observedOutcome: 0,
        });
      }

      const agg = byMonth.get(monthKey)!;
      agg.predictedAction += r.predictedAction;
      agg.counterfactualAction += r.counterfactualAction;
      agg.incrementalAction += r.incrementalAction;
      agg.predictedOutcome += r.predictedOutcome;
      agg.counterfactualOutcome += r.counterfactualOutcome;
      agg.incrementalOutcome += r.incrementalOutcome;
      agg.observedAction += r.observedAction;
      agg.observedOutcome += r.observedOutcome;
    }

    const summaries: MonthlySummary[] = [];
    for (const [month, agg] of byMonth) {
      // Lift % = (incremental / |counterfactual|) × 100
      const liftPercentAction =
        Math.abs(agg.counterfactualAction) > 0
          ? (agg.incrementalAction / Math.abs(agg.counterfactualAction)) * 100
          : 0;
      const liftPercentOutcome =
        Math.abs(agg.counterfactualOutcome) > 0
          ? (agg.incrementalOutcome / Math.abs(agg.counterfactualOutcome)) * 100
          : 0;

      summaries.push({
        month,
        totalPredictedAction: agg.predictedAction,
        totalCounterfactualAction: agg.counterfactualAction,
        totalIncrementalAction: agg.incrementalAction,
        totalPredictedOutcome: agg.predictedOutcome,
        totalCounterfactualOutcome: agg.counterfactualOutcome,
        totalIncrementalOutcome: agg.incrementalOutcome,
        totalObservedAction: agg.observedAction,
        totalObservedOutcome: agg.observedOutcome,
        liftPercentAction,
        liftPercentOutcome,
      });
    }

    // Sort by month ascending
    summaries.sort((a, b) => a.month.localeCompare(b.month));

    return summaries;
  }
}
