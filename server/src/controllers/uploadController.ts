/**
 * Upload Controller
 *
 * Handles CSV file uploads. Parses the CSV, validates rows,
 * and bulk-inserts observations into the database.
 */

import { Request, Response } from 'express';
import { v4 } from 'crypto';
import prisma from '../utils/prisma';
import { parseCSV } from '../utils/csvParser';

export async function uploadData(req: Request, res: Response): Promise<void> {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded. Send a CSV file as "file" field.' });
      return;
    }

    // Generate a batch ID for this upload
    const uploadBatchId = `batch_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;

    // Parse CSV buffer into typed observations
    const rawObservations = parseCSV(req.file.buffer);

    if (rawObservations.length === 0) {
      res.status(400).json({ error: 'CSV file is empty or has no valid rows.' });
      return;
    }

    // Bulk insert into database
    const created = await prisma.observation.createMany({
      data: rawObservations.map((obs) => ({
        hcpId: obs.hcpId,
        month: new Date(obs.month),
        suggestionType: obs.suggestionType,
        actionType: obs.actionType,
        outcomeType: obs.outcomeType,
        suggestionCount: obs.suggestionCount,
        actionCount: obs.actionCount,
        outcomeCount: obs.outcomeCount,
        specialtyCode: obs.specialtyCode || null,
        regionCode: obs.regionCode || null,
        tenureMonths: obs.tenureMonths ?? null,
        priorTrx: obs.priorTrx ?? null,
        priorNbrx: obs.priorNbrx ?? null,
        totalSuggestions: obs.totalSuggestions ?? null,
        theme: obs.theme || null,
        uploadBatchId,
      })),
    });

    // Gather distinct filter values for user reference
    const distinctSuggestionTypes = [...new Set(rawObservations.map((o) => o.suggestionType))];
    const distinctActionTypes = [...new Set(rawObservations.map((o) => o.actionType))];
    const distinctOutcomeTypes = [...new Set(rawObservations.map((o) => o.outcomeType))];

    res.json({
      success: true,
      uploadBatchId,
      rowsInserted: created.count,
      distinctValues: {
        suggestionTypes: distinctSuggestionTypes,
        actionTypes: distinctActionTypes,
        outcomeTypes: distinctOutcomeTypes,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Upload failed';
    res.status(500).json({ error: message });
  }
}
