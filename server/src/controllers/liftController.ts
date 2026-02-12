/**
 * Lift Controller
 *
 * Handles the full counterfactual simulation pipeline:
 *   1. Fetch observations
 *   2. Run Refinery (feature engineering + train + simulate)
 *   3. Persist lift results
 *   4. Return incremental lift data
 */

import { Request, Response } from 'express';
import prisma from '../utils/prisma';
import { Refinery } from '../services/refinery';
import { RunConfigSchema } from '../utils/validation';

export async function runLift(req: Request, res: Response): Promise<void> {
  try {
    // Validate request body
    const parsed = RunConfigSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: parsed.error.flatten() });
      return;
    }

    const config = parsed.data;

    // Fetch all observations
    const observations = await prisma.observation.findMany();
    if (observations.length === 0) {
      res.status(400).json({ error: 'No observations in database. Upload data first.' });
      return;
    }

    // Create run record
    const runId = `run_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
    await prisma.modelRun.create({
      data: {
        runId,
        status: 'simulating',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        adstockEnabled: config.adstockEnabled,
        adstockDecay: config.adstockDecay,
        theme: config.theme || null,
      },
    });

    // Run the full Refinery pipeline
    const refinery = new Refinery();
    let pipelineOutput;
    try {
      pipelineOutput = refinery.run(observations, config);
    } catch (pipelineError) {
      const msg = pipelineError instanceof Error ? pipelineError.message : 'Pipeline failed';
      await prisma.modelRun.update({
        where: { runId },
        data: { status: 'failed', errorMessage: msg },
      });
      res.status(400).json({ error: msg });
      return;
    }

    const { result, metrics } = pipelineOutput;

    // Persist lift results to database
    await prisma.liftResult.createMany({
      data: result.liftResults.map((lr) => ({
        runId,
        hcpId: lr.hcpId,
        month: new Date(lr.month),
        suggestionType: lr.suggestionType,
        actionType: lr.actionType,
        outcomeType: lr.outcomeType,
        theme: lr.theme || null,
        predictedAction: lr.predictedAction,
        counterfactualAction: lr.counterfactualAction,
        incrementalAction: lr.incrementalAction,
        predictedOutcome: lr.predictedOutcome,
        counterfactualOutcome: lr.counterfactualOutcome,
        incrementalOutcome: lr.incrementalOutcome,
        observedAction: lr.observedAction,
        observedOutcome: lr.observedOutcome,
      })),
    });

    // Persist model metadata
    const pathAMeta = await prisma.modelMetadata.create({
      data: {
        modelName: result.pathAModel.modelName,
        modelType: 'linear_regression',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        featureNames: result.pathAModel.featureNames,
        coefficients: result.pathAModel.coefficients,
        intercept: result.pathAModel.intercept,
        scalerMeans: result.pathAModel.scalerState.means,
        scalerStds: result.pathAModel.scalerState.stds,
        categoricalMap: result.pathAModel.categoricalMap as object,
        runId,
      },
    });

    // Persist feature importance for Path A
    await prisma.featureImportance.createMany({
      data: result.pathAModel.featureImportance.map((fi) => ({
        modelMetadataId: pathAMeta.id,
        featureName: fi.featureName,
        importance: fi.importance,
        rank: fi.rank,
      })),
    });

    const pathBMeta = await prisma.modelMetadata.create({
      data: {
        modelName: result.pathBModel.modelName,
        modelType: 'linear_regression',
        suggestionType: config.suggestionType,
        actionType: config.actionType,
        outcomeType: config.outcomeType,
        featureNames: result.pathBModel.featureNames,
        coefficients: result.pathBModel.coefficients,
        intercept: result.pathBModel.intercept,
        scalerMeans: result.pathBModel.scalerState.means,
        scalerStds: result.pathBModel.scalerState.stds,
        categoricalMap: result.pathBModel.categoricalMap as object,
        runId,
      },
    });

    // Persist feature importance for Path B
    await prisma.featureImportance.createMany({
      data: result.pathBModel.featureImportance.map((fi) => ({
        modelMetadataId: pathBMeta.id,
        featureName: fi.featureName,
        importance: fi.importance,
        rank: fi.rank,
      })),
    });

    // Update run status
    await prisma.modelRun.update({
      where: { runId },
      data: {
        status: 'completed',
        completedAt: new Date(),
        totalObservations: result.liftResults.length,
        rSquaredPathA: metrics.pathA.rSquared,
        rSquaredPathB: metrics.pathB.rSquared,
        maePathA: metrics.pathA.mae,
        maePathB: metrics.pathB.mae,
        modelMetadataId: pathAMeta.id,
      },
    });

    // Compute aggregate totals for the response
    const totalIncrementalAction = result.liftResults.reduce(
      (s, r) => s + r.incrementalAction,
      0
    );
    const totalIncrementalOutcome = result.liftResults.reduce(
      (s, r) => s + r.incrementalOutcome,
      0
    );
    const totalCounterfactualAction = result.liftResults.reduce(
      (s, r) => s + r.counterfactualAction,
      0
    );
    const overallLiftPercent =
      Math.abs(totalCounterfactualAction) > 0
        ? (totalIncrementalAction / Math.abs(totalCounterfactualAction)) * 100
        : 0;

    res.json({
      success: true,
      runId,
      totalObservations: result.liftResults.length,
      totalIncrementalAction,
      totalIncrementalOutcome,
      overallLiftPercent,
      monthlySummary: result.monthlySummary,
      metrics: {
        pathA: metrics.pathA,
        pathB: metrics.pathB,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Lift simulation failed';
    res.status(500).json({ error: message });
  }
}
