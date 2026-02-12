/**
 * Feature Importance Controller
 *
 * Returns ranked feature importance from trained models.
 * Supports filtering by runId and modelName.
 */

import { Request, Response } from 'express';
import prisma from '../utils/prisma';
import { FeatureImportanceQuerySchema } from '../utils/validation';

export async function getFeatureImportance(req: Request, res: Response): Promise<void> {
  try {
    const parsed = FeatureImportanceQuerySchema.safeParse(req.query);
    if (!parsed.success) {
      res.status(400).json({ error: parsed.error.flatten() });
      return;
    }

    const { runId, modelName } = parsed.data;

    // Build query filter for model metadata
    const metadataFilter: Record<string, unknown> = {};
    if (runId) metadataFilter.runId = runId;
    if (modelName) metadataFilter.modelName = modelName;

    // If no filters, get the most recent run
    let targetRunId = runId;
    if (!targetRunId) {
      const latestRun = await prisma.modelRun.findFirst({
        where: { status: 'completed' },
        orderBy: { completedAt: 'desc' },
      });
      if (latestRun) {
        targetRunId = latestRun.runId;
        metadataFilter.runId = targetRunId;
      }
    }

    // Fetch model metadata with feature importance
    const models = await prisma.modelMetadata.findMany({
      where: metadataFilter as any,
      include: {
        featureImportance: {
          orderBy: { rank: 'asc' },
        },
      },
    });

    if (models.length === 0) {
      res.status(404).json({
        error: 'No model metadata found. Train models first.',
      });
      return;
    }

    const result = models.map((model) => ({
      modelName: model.modelName,
      modelType: model.modelType,
      runId: model.runId,
      trainedAt: model.trainedAt,
      featureImportance: model.featureImportance.map((fi) => ({
        featureName: fi.featureName,
        importance: fi.importance,
        rank: fi.rank,
      })),
    }));

    res.json({
      runId: targetRunId,
      models: result,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to fetch feature importance';
    res.status(500).json({ error: message });
  }
}
