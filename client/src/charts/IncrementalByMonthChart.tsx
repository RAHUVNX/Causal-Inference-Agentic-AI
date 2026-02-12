/**
 * Incremental Lift by Month — Bar Chart
 *
 * Displays incremental action and incremental outcome per month.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface MonthlyData {
  month: string;
  totalIncrementalAction: number;
  totalIncrementalOutcome: number;
}

interface Props {
  data: MonthlyData[];
  outcomeLabel?: string;
}

export default function IncrementalByMonthChart({ data, outcomeLabel = 'Outcome' }: Props) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="month" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip
          contentStyle={{ fontSize: 13, borderRadius: 6 }}
          formatter={(value: number) => value.toFixed(2)}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar
          dataKey="totalIncrementalAction"
          name="Incremental Action"
          fill="#2563eb"
          radius={[4, 4, 0, 0]}
        />
        <Bar
          dataKey="totalIncrementalOutcome"
          name={`Incremental ${outcomeLabel}`}
          fill="#16a34a"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
