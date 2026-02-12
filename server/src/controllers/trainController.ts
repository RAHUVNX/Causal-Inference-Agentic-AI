/**
 * Train Controller
 *
 * Handles model training requests. Fetches observations from the database,
 * runs feature engineering, trains Path A and Path B models, and persists
 * model metadata and feature importance.
 */

import { Request, Response } from 'express';
import prisma from '../utils/prisma';
import { FeatureEngineer } from '../services/featureEngineer';
import { PathAActionModel, PathBOutcomeModel } from '../services/models';
import { TrainRequestSchema } from '../utils/validation';
import { FeatureEngineerConfig, TrainedModel } from '../utils/types';

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

    // Feature engineering
    const feConfig: FeatureEngineerConfig = {
      filter: {
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
      },
      adstockEnabled: config.adstockEnabled,
      adstockDecay: config.adstockDecay,
      theme: config.theme,
    };

    const featureEngineer = new FeatureEngineer(feConfig);
    let engineeringResult;
    try {
      engineeringResult = featureEngineer.engineer(observations);
    } catch (feError) {
      const msg = feError instanceof Error ? feError.message : 'Feature engineering failed';
      await prisma.modelRun.update({
        where: { runId },
        data: { status: 'failed', errorMessage: msg },
      });
      res.status(400).json({ error: msg });
      return;
    }

    const { rows, scalerState, categoricalMap, featureNames } = engineeringResult;

    // Train Path A: Suggestion → Action
    const pathAModel = new PathAActionModel();
    const pathAResult: TrainedModel = pathAModel.train(rows);
    pathAResult.scalerState = scalerState;
    pathAResult.categoricalMap = categoricalMap;

    // Persist Path A model metadata
    const pathAMeta = await prisma.modelMetadata.create({
      data: {
        modelName: pathAResult.modelName,
        modelType: 'linear_regression',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        featureNames,
        coefficients: pathAResult.coefficients,
        intercept: pathAResult.intercept,
        scalerMeans: scalerState.means,
        scalerStds: scalerState.stds,
        categoricalMap: categoricalMap as object,
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

    // Train Path B: Action → Outcome (TRX or NBRX)
    const pathBModel = new PathBOutcomeModel(config.outcomeType);
    const pathBResult: TrainedModel = pathBModel.train(rows);
    pathBResult.scalerState = scalerState;
    pathBResult.categoricalMap = categoricalMap;

    // Persist Path B model metadata
    const pathBMeta = await prisma.modelMetadata.create({
      data: {
        modelName: pathBResult.modelName,
        modelType: 'linear_regression',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        featureNames,
        coefficients: pathBResult.coefficients,
        intercept: pathBResult.intercept,
        scalerMeans: scalerState.means,
        scalerStds: scalerState.stds,
        categoricalMap: categoricalMap as object,
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
        totalObservations: rows.length,
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
      totalObservations: rows.length,
      pathA: {
        modelName: pathAResult.modelName,
        rSquared: pathAResult.rSquared,
        mae: pathAResult.mae,
        featureCount: featureNames.length,
      },
      pathB: {
        modelName: pathBResult.modelName,
        rSquared: pathBResult.rSquared,
        mae: pathBResult.mae,
        featureCount: featureNames.length,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Training failed';
    res.status(500).json({ error: message });
  }
}
