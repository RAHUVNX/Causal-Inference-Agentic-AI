/**
 * Summary Controller
 *
 * Returns monthly aggregated lift results and overall KPIs.
 * Queries persisted LiftResult records and aggregates them by month.
 */

import { Request, Response } from 'express';
import prisma from '../utils/prisma';
import { SummaryQuerySchema } from '../utils/validation';

export async function getSummary(req: Request, res: Response): Promise<void> {
  try {
    const parsed = SummaryQuerySchema.safeParse(req.query);
    if (!parsed.success) {
      res.status(400).json({ error: parsed.error.flatten() });
      return;
    }

    // If runId is specified, filter by it; otherwise get the most recent run
    let runId = parsed.data.runId;

    if (!runId) {
      const latestRun = await prisma.modelRun.findFirst({
        where: { status: 'completed' },
        orderBy: { completedAt: 'desc' },
      });

      if (!latestRun) {
        res.status(404).json({ error: 'No completed runs found. Run lift simulation first.' });
        return;
      }
      runId = latestRun.runId;
    }

    // Fetch lift results for this run
    const liftResults = await prisma.liftResult.findMany({
      where: { runId },
      orderBy: { month: 'asc' },
    });

    if (liftResults.length === 0) {
      res.status(404).json({ error: `No lift results found for run ${runId}.` });
      return;
    }

    // Fetch model run metadata
    const modelRun = await prisma.modelRun.findUnique({
      where: { runId },
    });

    // Aggregate by month
    const byMonth = new Map<string, {
      predictedAction: number;
      counterfactualAction: number;
      incrementalAction: number;
      predictedOutcome: number;
      counterfactualOutcome: number;
      incrementalOutcome: number;
      observedAction: number;
      observedOutcome: number;
      count: number;
    }>();

    for (const lr of liftResults) {
      const monthKey = lr.month.toISOString().slice(0, 7);
      if (!byMonth.has(monthKey)) {
        byMonth.set(monthKey, {
          predictedAction: 0,
          counterfactualAction: 0,
          incrementalAction: 0,
          predictedOutcome: 0,
          counterfactualOutcome: 0,
          incrementalOutcome: 0,
          observedAction: 0,
          observedOutcome: 0,
          count: 0,
        });
      }
      const agg = byMonth.get(monthKey)!;
      agg.predictedAction += lr.predictedAction;
      agg.counterfactualAction += lr.counterfactualAction;
      agg.incrementalAction += lr.incrementalAction;
      agg.predictedOutcome += lr.predictedOutcome;
      agg.counterfactualOutcome += lr.counterfactualOutcome;
      agg.incrementalOutcome += lr.incrementalOutcome;
      agg.observedAction += lr.observedAction;
      agg.observedOutcome += lr.observedOutcome;
      agg.count++;
    }

    const monthlySummary = Array.from(byMonth.entries())
      .map(([month, agg]) => ({
        month,
        totalPredictedAction: agg.predictedAction,
        totalCounterfactualAction: agg.counterfactualAction,
        totalIncrementalAction: agg.incrementalAction,
        totalPredictedOutcome: agg.predictedOutcome,
        totalCounterfactualOutcome: agg.counterfactualOutcome,
        totalIncrementalOutcome: agg.incrementalOutcome,
        totalObservedAction: agg.observedAction,
        totalObservedOutcome: agg.observedOutcome,
        liftPercentAction:
          Math.abs(agg.counterfactualAction) > 0
            ? (agg.incrementalAction / Math.abs(agg.counterfactualAction)) * 100
            : 0,
        liftPercentOutcome:
          Math.abs(agg.counterfactualOutcome) > 0
            ? (agg.incrementalOutcome / Math.abs(agg.counterfactualOutcome)) * 100
            : 0,
        hcpCount: agg.count,
      }))
      .sort((a, b) => a.month.localeCompare(b.month));

    // Overall totals
    const totalIncrementalAction = liftResults.reduce((s, r) => s + r.incrementalAction, 0);
    const totalIncrementalOutcome = liftResults.reduce((s, r) => s + r.incrementalOutcome, 0);
    const totalCounterfactualAction = liftResults.reduce((s, r) => s + r.counterfactualAction, 0);
    const totalCounterfactualOutcome = liftResults.reduce((s, r) => s + r.counterfactualOutcome, 0);
    const totalObservedAction = liftResults.reduce((s, r) => s + r.observedAction, 0);
    const totalObservedOutcome = liftResults.reduce((s, r) => s + r.observedOutcome, 0);
    const totalPredictedAction = liftResults.reduce((s, r) => s + r.predictedAction, 0);
    const totalPredictedOutcome = liftResults.reduce((s, r) => s + r.predictedOutcome, 0);

    const overallLiftPercentAction =
      Math.abs(totalCounterfactualAction) > 0
        ? (totalIncrementalAction / Math.abs(totalCounterfactualAction)) * 100
        : 0;

    const overallLiftPercentOutcome =
      Math.abs(totalCounterfactualOutcome) > 0
        ? (totalIncrementalOutcome / Math.abs(totalCounterfactualOutcome)) * 100
        : 0;

    res.json({
      runId,
      outcomeType: modelRun?.outcomeType || liftResults[0].outcomeType,
      totalObservations: liftResults.length,
      kpis: {
        totalIncrementalAction,
        totalIncrementalOutcome,
        totalObservedAction,
        totalObservedOutcome,
        totalPredictedAction,
        totalPredictedOutcome,
        overallLiftPercentAction,
        overallLiftPercentOutcome,
      },
      monthlySummary,
      modelRun: modelRun
        ? {
            status: modelRun.status,
            rSquaredPathA: modelRun.rSquaredPathA,
            rSquaredPathB: modelRun.rSquaredPathB,
            maePathA: modelRun.maePathA,
            maePathB: modelRun.maePathB,
            startedAt: modelRun.startedAt,
            completedAt: modelRun.completedAt,
          }
        : null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to fetch summary';
    res.status(500).json({ error: message });
  }
}
