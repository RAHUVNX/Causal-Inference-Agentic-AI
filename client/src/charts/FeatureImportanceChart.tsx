/**
 * Feature Importance — Horizontal Bar Chart
 *
 * Displays ranked feature importance from trained models.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface FeatureItem {
  featureName: string;
  importance: number;
  rank: number;
}

interface Props {
  data: FeatureItem[];
  title?: string;
  color?: string;
  maxFeatures?: number;
}

export default function FeatureImportanceChart({
  data,
  title,
  color = '#2563eb',
  maxFeatures = 15,
}: Props) {
  // Take top N features sorted by rank
  const chartData = data
    .slice(0, maxFeatures)
    .sort((a, b) => b.importance - a.importance)
    .map((item) => ({
      name: item.featureName,
      importance: parseFloat(item.importance.toFixed(4)),
    }));

  return (
    <div>
      {title && <h3 style={{ fontSize: 14, marginBottom: 12 }}>{title}</h3>}
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 28)}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 20, bottom: 5, left: 120 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11 }}
            width={110}
          />
          <Tooltip
            contentStyle={{ fontSize: 13, borderRadius: 6 }}
            formatter={(value: number) => value.toFixed(4)}
          />
          <Bar dataKey="importance" fill={color} radius={[0, 4, 4, 0]} barSize={18} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
