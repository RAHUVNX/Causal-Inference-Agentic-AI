/**
 * Shared types for the Causal Lift Web Platform.
 *
 * The platform implements a two-path causal framework:
 *   Path A: Suggestion → Action
 *   Path B: Action → Outcome (TRX / NBRX)
 *
 * Incremental lift is computed as:
 *   incremental_action  = predicted_action  − counterfactual_action
 *   incremental_outcome = predicted_outcome − counterfactual_outcome
 */

// ── Filter Triangle ────────────────────────────────────────
export interface FilterTriangle {
  suggestionType: string;
  actionType: string;
  outcomeType: 'trx' | 'nbrx';
}

// ── Raw row from CSV ───────────────────────────────────────
export interface RawObservation {
  hcpId: string;
  month: string; // ISO date string
  suggestionType: string;
  actionType: string;
  outcomeType: string;
  suggestionCount: number;
  actionCount: number;
  outcomeCount: number;
  specialtyCode?: string;
  regionCode?: string;
  tenureMonths?: number;
  priorTrx?: number;
  priorNbrx?: number;
  totalSuggestions?: number;
  theme?: string;
}

// ── Engineered feature row ─────────────────────────────────
export interface EngineeredRow {
  hcpId: string;
  month: string;
  features: number[];        // Numeric feature vector after encoding + scaling
  featureNames: string[];    // Corresponding feature names
  actionCount: number;       // Ground truth for Path A
  outcomeCount: number;      // Ground truth for Path B (TRX or NBRX)
  suggestionCount: number;   // Original suggestion value (needed for counterfactual zeroing)
  observedAction: number;
  observedOutcome: number;
  theme?: string;
}

// ── Feature engineering config ─────────────────────────────
export interface FeatureEngineerConfig {
  filter: FilterTriangle;
  adstockEnabled: boolean;
  adstockDecay: number;      // λ ∈ (0, 1) for geometric adstock
  studyPeriodStart?: string; // ISO date
  studyPeriodEnd?: string;   // ISO date
  theme?: string;
}

// ── Scaler state for standardization ───────────────────────
export interface ScalerState {
  means: Record<string, number>;
  stds: Record<string, number>;
}

// ── Categorical encoding map ───────────────────────────────
export interface CategoricalMap {
  [column: string]: { [category: string]: number[] }; // one-hot vectors
}

// ── Trained model representation ───────────────────────────
export interface TrainedModel {
  modelName: string;               // "PathA_Action" | "PathB_TRX" | "PathB_NBRX"
  coefficients: number[];          // β weights for linear regression
  intercept: number;               // β₀
  featureNames: string[];
  scalerState: ScalerState;
  categoricalMap: CategoricalMap;
  featureImportance: FeatureImportanceEntry[];
  rSquared: number;
  mae: number;
}

export interface FeatureImportanceEntry {
  featureName: string;
  importance: number;
  rank: number;
}

// ── Lift result for a single HCP-month ─────────────────────
export interface LiftResultRow {
  hcpId: string;
  month: string;
  suggestionType: string;
  actionType: string;
  outcomeType: string;
  theme?: string;
  predictedAction: number;
  counterfactualAction: number;
  incrementalAction: number;
  predictedOutcome: number;
  counterfactualOutcome: number;
  incrementalOutcome: number;
  observedAction: number;
  observedOutcome: number;
}

// ── Aggregate summary row ──────────────────────────────────
export interface MonthlySummary {
  month: string;
  totalPredictedAction: number;
  totalCounterfactualAction: number;
  totalIncrementalAction: number;
  totalPredictedOutcome: number;
  totalCounterfactualOutcome: number;
  totalIncrementalOutcome: number;
  totalObservedAction: number;
  totalObservedOutcome: number;
  liftPercentAction: number;
  liftPercentOutcome: number;
}

// ── Run configuration ──────────────────────────────────────
export interface RunConfig {
  suggestionType: string;
  actionType: string;
  outcomeType: 'trx' | 'nbrx';
  adstockEnabled?: boolean;
  adstockDecay?: number;
  theme?: string;
}

// ── Full pipeline result ───────────────────────────────────
export interface PipelineResult {
  runId: string;
  pathAModel: TrainedModel;
  pathBModel: TrainedModel;
  liftResults: LiftResultRow[];
  monthlySummary: MonthlySummary[];
}
