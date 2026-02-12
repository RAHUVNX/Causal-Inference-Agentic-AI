interface KpiCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  variant?: 'default' | 'positive' | 'negative';
}

export default function KpiCard({ label, value, subtitle, variant = 'default' }: KpiCardProps) {
  return (
    <div className={`kpi-card ${variant}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}</div>
      {subtitle && <div className="kpi-sub">{subtitle}</div>}
    </div>
  );
}
