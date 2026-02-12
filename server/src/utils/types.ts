/**
 * Shared types for the Causal Lift Web Platform.
 *
 * The platform implements a two-path causal framework:
 *   Path A: Suggestion -> Action
 *   Path B: Action -> Outcome (TRX / NBRX)
 *
 * ML computation is handled by the Python service.
 * These types are used for database operations and API responses.
 */

// -- Raw row from CSV ---
export interface RawObservation {
  hcpId: string;
  month: string;
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

// -- Run configuration ---
export interface RunConfig {
  suggestionType: string;
  actionType: string;
  outcomeType: 'trx' | 'nbrx';
  adstockEnabled?: boolean;
  adstockDecay?: number;
  theme?: string;
}
