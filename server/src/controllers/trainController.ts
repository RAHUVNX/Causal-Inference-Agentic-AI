/**
 * Train Controller
 *
 * Handles model training requests. Fetches observations from the database,
 * sends them to the Python ML service for training, and persists
 * model metadata and feature importance.
 */

import { Request, Response } from 'express';
import prisma from '../utils/prisma';
import { callPythonTrain } from '../utils/pythonService';
import { TrainRequestSchema } from '../utils/validation';

export async function trainModels(req: Request, res: Response): Promise<void> {
  try {
    // Validate request body
    const parsed = TrainRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: parsed.error.flatten() });
      return;
    }

    const config = parsed.data;
    const runId = `run_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;

    // Create a model run record to track status
    await prisma.modelRun.create({
      data: {
        runId,
        status: 'training',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        adstockEnabled: config.adstockEnabled,
        adstockDecay: config.adstockDecay,
        theme: config.theme || null,
      },
    });

    // Fetch all observations from the database
    const observations = await prisma.observation.findMany();

    if (observations.length === 0) {
      await prisma.modelRun.update({
        where: { runId },
        data: { status: 'failed', errorMessage: 'No observations in database' },
      });
      res.status(400).json({ error: 'No observations in database. Upload data first.' });
      return;
    }

    // Convert observations for Python service
    const obsPayload = observations.map((obs) => ({
      hcpId: obs.hcpId,
      month: obs.month.toISOString(),
      suggestionType: obs.suggestionType,
      actionType: obs.actionType,
      outcomeType: obs.outcomeType,
      suggestionCount: obs.suggestionCount,
      actionCount: obs.actionCount,
      outcomeCount: obs.outcomeCount,
      specialtyCode: obs.specialtyCode,
      regionCode: obs.regionCode,
      tenureMonths: obs.tenureMonths,
      priorTrx: obs.priorTrx,
      priorNbrx: obs.priorNbrx,
      totalSuggestions: obs.totalSuggestions,
      theme: obs.theme,
    }));

    // Call Python ML service for training
    let trainResult;
    try {
      trainResult = await callPythonTrain({
        observations: obsPayload,
        filter: {
          suggestionType: config.suggestionType,
          actionType: config.actionType,
          outcomeType: config.outcomeType,
        },
        adstockEnabled: config.adstockEnabled,
        adstockDecay: config.adstockDecay,
        theme: config.theme,
      });
    } catch (pyError: any) {
      const msg = pyError.response?.data?.detail || pyError.message || 'Python ML service error';
      await prisma.modelRun.update({
        where: { runId },
        data: { status: 'failed', errorMessage: msg },
      });
      res.status(400).json({ error: msg });
      return;
    }

    const { pathA: pathAResult, pathB: pathBResult } = trainResult;

    // Persist Path A model metadata
    const pathAMeta = await prisma.modelMetadata.create({
      data: {
        modelName: pathAResult.modelName,
        modelType: 'linear_regression',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        featureNames: pathAResult.featureNames,
        coefficients: pathAResult.coefficients,
        intercept: pathAResult.intercept,
        scalerMeans: pathAResult.scalerMeans,
        scalerStds: pathAResult.scalerStds,
        categoricalMap: pathAResult.categoricalMap as object,
        runId,
      },
    });

    // Persist Path A feature importance
    await prisma.featureImportance.createMany({
      data: pathAResult.featureImportance.map((fi) => ({
        modelMetadataId: pathAMeta.id,
        featureName: fi.featureName,
        importance: fi.importance,
        rank: fi.rank,
      })),
    });

    // Persist Path B model metadata
    const pathBMeta = await prisma.modelMetadata.create({
      data: {
        modelName: pathBResult.modelName,
        modelType: 'linear_regression',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        featureNames: pathBResult.featureNames,
        coefficients: pathBResult.coefficients,
        intercept: pathBResult.intercept,
        scalerMeans: pathBResult.scalerMeans,
        scalerStds: pathBResult.scalerStds,
        categoricalMap: pathBResult.categoricalMap as object,
        runId,
      },
    });

    // Persist Path B feature importance
    await prisma.featureImportance.createMany({
      data: pathBResult.featureImportance.map((fi) => ({
        modelMetadataId: pathBMeta.id,
        featureName: fi.featureName,
        importance: fi.importance,
        rank: fi.rank,
      })),
    });

    // Update model run status
    await prisma.modelRun.update({
      where: { runId },
      data: {
        status: 'completed',
        completedAt: new Date(),
        totalObservations: observations.length,
        rSquaredPathA: pathAResult.rSquared,
        rSquaredPathB: pathBResult.rSquared,
        maePathA: pathAResult.mae,
        maePathB: pathBResult.mae,
        modelMetadataId: pathAMeta.id,
      },
    });

    res.json({
      success: true,
      runId,
      totalObservations: observations.length,
      pathA: {
        modelName: pathAResult.modelName,
        rSquared: pathAResult.rSquared,
        mae: pathAResult.mae,
        featureCount: pathAResult.featureNames.length,
      },
      pathB: {
        modelName: pathBResult.modelName,
        rSquared: pathBResult.rSquared,
        mae: pathBResult.mae,
        featureCount: pathBResult.featureNames.length,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Training failed';
    res.status(500).json({ error: message });
  }
}
