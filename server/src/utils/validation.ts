/**
 * Zod validation schemas for API request payloads.
 */
import { z } from 'zod';

export const RunConfigSchema = z.object({
  suggestionType: z.string().min(1, 'suggestionType is required'),
  actionType: z.string().min(1, 'actionType is required'),
  outcomeType: z.enum(['trx', 'nbrx'], {
    errorMap: () => ({ message: 'outcomeType must be "trx" or "nbrx"' }),
  }),
  adstockEnabled: z.boolean().optional().default(false),
  adstockDecay: z.number().min(0).max(1).optional().default(0.5),
  theme: z.string().optional(),
});

export const TrainRequestSchema = z.object({
  suggestionType: z.string().min(1),
  actionType: z.string().min(1),
  outcomeType: z.enum(['trx', 'nbrx']),
  adstockEnabled: z.boolean().optional().default(false),
  adstockDecay: z.number().min(0).max(1).optional().default(0.5),
  theme: z.string().optional(),
});

export const SummaryQuerySchema = z.object({
  runId: z.string().optional(),
});

export const FeatureImportanceQuerySchema = z.object({
  runId: z.string().optional(),
  modelName: z.string().optional(),
});
