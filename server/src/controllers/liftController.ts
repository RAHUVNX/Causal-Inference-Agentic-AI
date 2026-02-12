/**
 * Lift Controller
 *
 * Handles the full counterfactual simulation pipeline:
 *   1. Fetch observations from database
 *   2. Send to Python ML service for training + simulation
 *   3. Persist lift results and model metadata
 *   4. Return incremental lift data
 */

import { Request, Response } from 'express';
import prisma from '../utils/prisma';
import { callPythonRunLift } from '../utils/pythonService';
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

    // Call Python ML service for full pipeline
    let pipelineOutput;
    try {
      pipelineOutput = await callPythonRunLift({
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

    const { liftResults, monthlySummary, metrics, pathA: pathAResult, pathB: pathBResult } = pipelineOutput;

    // Persist lift results to database
    await prisma.liftResult.createMany({
      data: liftResults.map((lr) => ({
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

    // Persist feature importance for Path A
    await prisma.featureImportance.createMany({
      data: pathAResult.featureImportance.map((fi) => ({
        modelMetadataId: pathAMeta.id,
        featureName: fi.featureName,
        importance: fi.importance,
        rank: fi.rank,
      })),
    });

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

    // Persist feature importance for Path B
    await prisma.featureImportance.createMany({
      data: pathBResult.featureImportance.map((fi) => ({
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
        totalObservations: liftResults.length,
        rSquaredPathA: metrics.pathA.rSquared,
        rSquaredPathB: metrics.pathB.rSquared,
        maePathA: metrics.pathA.mae,
        maePathB: metrics.pathB.mae,
        modelMetadataId: pathAMeta.id,
      },
    });

    // Compute aggregate totals for the response
    const totalIncrementalAction = liftResults.reduce(
      (s, r) => s + r.incrementalAction,
      0
    );
    const totalIncrementalOutcome = liftResults.reduce(
      (s, r) => s + r.incrementalOutcome,
      0
    );
    const totalCounterfactualAction = liftResults.reduce(
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
      totalObservations: liftResults.length,
      totalIncrementalAction,
      totalIncrementalOutcome,
      overallLiftPercent,
      monthlySummary,
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
