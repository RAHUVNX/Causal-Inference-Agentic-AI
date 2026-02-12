/**
 * HTTP client for calling the Python ML service.
 */

import axios from 'axios';

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5001';

const client = axios.create({
  baseURL: PYTHON_SERVICE_URL,
  timeout: 300000, // 5 minutes for long-running ML operations
  headers: { 'Content-Type': 'application/json' },
});

export async function callPythonTrain(payload: {
  observations: object[];
  filter: { suggestionType: string; actionType: string; outcomeType: string };
  adstockEnabled: boolean;
  adstockDecay: number;
  theme?: string;
}): Promise<{
  pathA: TrainedModelResult;
  pathB: TrainedModelResult;
}> {
  const { data } = await client.post('/train', payload);
  return data;
}

export async function callPythonRunLift(payload: {
  observations: object[];
  filter: { suggestionType: string; actionType: string; outcomeType: string };
  adstockEnabled: boolean;
  adstockDecay: number;
  theme?: string;
}): Promise<{
  liftResults: LiftResultItem[];
  monthlySummary: MonthlySummaryItem[];
  metrics: { pathA: MetricsItem; pathB: MetricsItem };
  pathA: TrainedModelResult;
  pathB: TrainedModelResult;
}> {
  const { data } = await client.post('/run-lift', payload);
  return data;
}

export async function checkPythonHealth(): Promise<boolean> {
  try {
    const { data } = await client.get('/health');
    return data.status === 'ok';
  } catch {
    return false;
  }
}

// Types matching Python service responses
export interface TrainedModelResult {
  modelName: string;
  coefficients: number[];
  intercept: number;
  featureNames: string[];
  scalerMeans: Record<string, number>;
  scalerStds: Record<string, number>;
  categoricalMap: Record<string, Record<string, number[]>>;
  featureImportance: { featureName: string; importance: number; rank: number }[];
  rSquared: number;
  mae: number;
}

export interface LiftResultItem {
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

export interface MonthlySummaryItem {
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

export interface MetricsItem {
  rSquared: number;
  mae: number;
  rmse: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}
