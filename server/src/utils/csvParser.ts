/**
 * CSV parsing utility.
 *
 * Parses uploaded CSV files into RawObservation objects,
 * performing type coercion and validation on numeric fields.
 */

import { parse } from 'csv-parse/sync';
import { RawObservation } from './types';

/**
 * Column name mappings from CSV headers to internal field names.
 * Supports snake_case CSV headers → camelCase internal names.
 */
const COLUMN_MAP: Record<string, keyof RawObservation> = {
  hcp_id: 'hcpId',
  hcpid: 'hcpId',
  month: 'month',
  suggestion_type: 'suggestionType',
  suggestiontype: 'suggestionType',
  action_type: 'actionType',
  actiontype: 'actionType',
  outcome_type: 'outcomeType',
  outcometype: 'outcomeType',
  suggestion_count: 'suggestionCount',
  suggestioncount: 'suggestionCount',
  action_count: 'actionCount',
  actioncount: 'actionCount',
  outcome_count: 'outcomeCount',
  outcomecount: 'outcomeCount',
  specialty_code: 'specialtyCode',
  specialtycode: 'specialtyCode',
  region_code: 'regionCode',
  regioncode: 'regionCode',
  tenure_months: 'tenureMonths',
  tenuremonths: 'tenureMonths',
  prior_trx: 'priorTrx',
  priortrx: 'priorTrx',
  prior_nbrx: 'priorNbrx',
  priornbrx: 'priorNbrx',
  total_suggestions: 'totalSuggestions',
  totalsuggestions: 'totalSuggestions',
  theme: 'theme',
};

const NUMERIC_FIELDS = new Set([
  'suggestionCount',
  'actionCount',
  'outcomeCount',
  'tenureMonths',
  'priorTrx',
  'priorNbrx',
  'totalSuggestions',
]);

/**
 * Parse a CSV buffer into an array of RawObservation objects.
 */
export function parseCSV(buffer: Buffer): RawObservation[] {
  const records = parse(buffer, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });

  return records.map((record: Record<string, string>) => {
    const obs: Partial<RawObservation> = {};

    for (const [csvCol, value] of Object.entries(record)) {
      const normalizedCol = csvCol.toLowerCase().trim();
      const fieldName = COLUMN_MAP[normalizedCol];

      if (!fieldName) continue;

      if (NUMERIC_FIELDS.has(fieldName)) {
        (obs as Record<string, unknown>)[fieldName] = parseFloat(value) || 0;
      } else {
        (obs as Record<string, unknown>)[fieldName] = value;
      }
    }

    // Validate required fields
    if (!obs.hcpId || !obs.month || !obs.suggestionType || !obs.actionType || !obs.outcomeType) {
      throw new Error(
        `Missing required fields. Row must have: hcp_id, month, suggestion_type, action_type, outcome_type. Got: ${JSON.stringify(record)}`
      );
    }

    return {
      hcpId: obs.hcpId!,
      month: obs.month!,
      suggestionType: obs.suggestionType!,
      actionType: obs.actionType!,
      outcomeType: obs.outcomeType!,
      suggestionCount: obs.suggestionCount || 0,
      actionCount: obs.actionCount || 0,
      outcomeCount: obs.outcomeCount || 0,
      specialtyCode: obs.specialtyCode,
      regionCode: obs.regionCode,
      tenureMonths: obs.tenureMonths,
      priorTrx: obs.priorTrx,
      priorNbrx: obs.priorNbrx,
      totalSuggestions: obs.totalSuggestions,
      theme: obs.theme,
    };
  });
}
