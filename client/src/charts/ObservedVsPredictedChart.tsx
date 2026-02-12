/**
 * Observed vs Predicted — Line Chart
 *
 * Compares observed values against model predictions across months.
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface MonthlyData {
  month: string;
  totalObservedOutcome: number;
  totalPredictedOutcome: number;
  totalObservedAction: number;
  totalPredictedAction: number;
}

interface Props {
  data: MonthlyData[];
  type: 'action' | 'outcome';
  label?: string;
}

export default function ObservedVsPredictedChart({ data, type, label }: Props) {
  const observedKey = type === 'action' ? 'totalObservedAction' : 'totalObservedOutcome';
  const predictedKey = type === 'action' ? 'totalPredictedAction' : 'totalPredictedOutcome';
  const typeLabel = label || (type === 'action' ? 'Action' : 'Outcome');

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="month" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip
          contentStyle={{ fontSize: 13, borderRadius: 6 }}
          formatter={(value: number) => value.toFixed(2)}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey={observedKey}
          name={`Observed ${typeLabel}`}
          stroke="#64748b"
          strokeWidth={2}
          dot={{ r: 3 }}
        />
        <Line
          type="monotone"
          dataKey={predictedKey}
          name={`Predicted ${typeLabel}`}
          stroke="#2563eb"
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={{ r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
