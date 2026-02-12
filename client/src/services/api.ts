/**
 * API service layer.
 * All HTTP requests to the backend are centralized here.
 */

import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 minutes for long-running pipeline operations
});

// ── Types matching backend responses ─────────────────────

export interface UploadResponse {
  success: boolean;
  uploadBatchId: string;
  rowsInserted: number;
  distinctValues: {
    suggestionTypes: string[];
    actionTypes: string[];
    outcomeTypes: string[];
  };
}

export interface TrainRequest {
  suggestionType: string;
  actionType: string;
  outcomeType: 'trx' | 'nbrx';
  adstockEnabled?: boolean;
  adstockDecay?: number;
  theme?: string;
}

export interface TrainResponse {
  success: boolean;
  runId: string;
  totalObservations: number;
  pathA: { modelName: string; rSquared: number; mae: number; featureCount: number };
  pathB: { modelName: string; rSquared: number; mae: number; featureCount: number };
}

export interface LiftRequest {
  suggestionType: string;
  actionType: string;
  outcomeType: 'trx' | 'nbrx';
  adstockEnabled?: boolean;
  adstockDecay?: number;
  theme?: string;
}

export interface MonthlySummaryRow {
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

export interface ModelMetrics {
  rSquared: number;
  mae: number;
  rmse: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface LiftResponse {
  success: boolean;
  runId: string;
  totalObservations: number;
  totalIncrementalAction: number;
  totalIncrementalOutcome: number;
  overallLiftPercent: number;
  monthlySummary: MonthlySummaryRow[];
  metrics: {
    pathA: ModelMetrics;
    pathB: ModelMetrics;
  };
}

export interface SummaryResponse {
  runId: string;
  outcomeType: string;
  totalObservations: number;
  kpis: {
    totalIncrementalAction: number;
    totalIncrementalOutcome: number;
    totalObservedAction: number;
    totalObservedOutcome: number;
    totalPredictedAction: number;
    totalPredictedOutcome: number;
    overallLiftPercentAction: number;
    overallLiftPercentOutcome: number;
  };
  monthlySummary: (MonthlySummaryRow & { hcpCount: number })[];
  modelRun: {
    status: string;
    rSquaredPathA: number | null;
    rSquaredPathB: number | null;
    maePathA: number | null;
    maePathB: number | null;
    startedAt: string;
    completedAt: string | null;
  } | null;
}

export interface FeatureImportanceItem {
  featureName: string;
  importance: number;
  rank: number;
}

export interface FeatureImportanceResponse {
  runId: string;
  models: {
    modelName: string;
    modelType: string;
    runId: string;
    trainedAt: string;
    featureImportance: FeatureImportanceItem[];
  }[];
}

// ── API methods ──────────────────────────────────────────

export async function uploadData(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload-data', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function trainModels(params: TrainRequest): Promise<TrainResponse> {
  const { data } = await api.post<TrainResponse>('/train-models', params);
  return data;
}

export async function runLift(params: LiftRequest): Promise<LiftResponse> {
  const { data } = await api.post<LiftResponse>('/run-lift', params);
  return data;
}

export async function getSummary(runId?: string): Promise<SummaryResponse> {
  const params = runId ? { runId } : {};
  const { data } = await api.get<SummaryResponse>('/summary', { params });
  return data;
}

export async function getFeatureImportance(
  runId?: string,
  modelName?: string
): Promise<FeatureImportanceResponse> {
  const params: Record<string, string> = {};
  if (runId) params.runId = runId;
  if (modelName) params.modelName = modelName;
  const { data } = await api.get<FeatureImportanceResponse>('/feature-importance', { params });
  return data;
}
