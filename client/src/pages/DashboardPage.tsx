import { useState, useEffect } from 'react';
import { getSummary, getFeatureImportance, SummaryResponse, FeatureImportanceResponse } from '../services/api';
import KpiCard from '../components/KpiCard';
import LoadingSpinner from '../components/LoadingSpinner';
import IncrementalByMonthChart from '../charts/IncrementalByMonthChart';
import ObservedVsPredictedChart from '../charts/ObservedVsPredictedChart';
import FeatureImportanceChart from '../charts/FeatureImportanceChart';

export default function DashboardPage() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [featureData, setFeatureData] = useState<FeatureImportanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDashboard() {
      setLoading(true);
      setError(null);
      try {
        const [summaryRes, fiRes] = await Promise.all([
          getSummary(),
          getFeatureImportance(),
        ]);
        setSummary(summaryRes);
        setFeatureData(fiRes);
      } catch (err: any) {
        // If no data yet, show a friendly message
        if (err.response?.status === 404) {
          setError('No completed simulation runs yet. Upload data and run a lift simulation to see the dashboard.');
        } else {
          setError(err.response?.data?.error || err.message || 'Failed to load dashboard');
        }
      } finally {
        setLoading(false);
      }
    }
    loadDashboard();
  }, []);

  if (loading) return <LoadingSpinner message="Loading dashboard data..." />;

  if (error) {
    return (
      <div>
        <div className="page-header">
          <h2>Dashboard</h2>
          <p>Causal lift analysis overview with KPIs, charts, and model insights.</p>
        </div>
        <div className="alert alert-warning">{error}</div>
      </div>
    );
  }

  if (!summary) return null;

  const { kpis, monthlySummary, modelRun } = summary;
  const outcomeLabel = summary.outcomeType === 'nbrx' ? 'NBRX' : 'TRX';

  // Find Path A and Path B feature importance
  const pathAFeatures = featureData?.models.find((m) =>
    m.modelName.includes('PathA')
  )?.featureImportance;
  const pathBFeatures = featureData?.models.find((m) =>
    m.modelName.includes('PathB')
  )?.featureImportance;

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>
          Causal lift analysis for run <strong>{summary.runId}</strong> &mdash;{' '}
          {summary.totalObservations} observations
        </p>
      </div>

      {/* ── KPI Cards ──────────────────────────────── */}
      <div className="kpi-grid">
        <KpiCard
          label={`Total Incremental ${outcomeLabel}`}
          value={kpis.totalIncrementalOutcome}
          variant={kpis.totalIncrementalOutcome > 0 ? 'positive' : 'negative'}
          subtitle="Predicted - Counterfactual"
        />
        <KpiCard
          label="Total Incremental Action"
          value={kpis.totalIncrementalAction}
          variant={kpis.totalIncrementalAction > 0 ? 'positive' : 'negative'}
          subtitle="Predicted - Counterfactual"
        />
        <KpiCard
          label={`Lift % (${outcomeLabel})`}
          value={`${kpis.overallLiftPercentOutcome.toFixed(2)}%`}
          variant={kpis.overallLiftPercentOutcome > 0 ? 'positive' : 'negative'}
          subtitle="Incr / |Counterfactual|"
        />
        <KpiCard
          label="Lift % (Action)"
          value={`${kpis.overallLiftPercentAction.toFixed(2)}%`}
          variant={kpis.overallLiftPercentAction > 0 ? 'positive' : 'negative'}
          subtitle="Incr / |Counterfactual|"
        />
      </div>

      {/* ── Model quality indicators ────────────────── */}
      {modelRun && (
        <div className="kpi-grid">
          <KpiCard
            label="R² Path A"
            value={modelRun.rSquaredPathA?.toFixed(4) || 'N/A'}
            subtitle="Suggestion → Action"
          />
          <KpiCard
            label="R² Path B"
            value={modelRun.rSquaredPathB?.toFixed(4) || 'N/A'}
            subtitle={`Action → ${outcomeLabel}`}
          />
          <KpiCard
            label="MAE Path A"
            value={modelRun.maePathA?.toFixed(4) || 'N/A'}
            subtitle="Mean Absolute Error"
          />
          <KpiCard
            label="MAE Path B"
            value={modelRun.maePathB?.toFixed(4) || 'N/A'}
            subtitle="Mean Absolute Error"
          />
        </div>
      )}

      {/* ── Charts ─────────────────────────────────── */}
      <div className="chart-grid">
        <div className="chart-card">
          <h3>Observed vs Predicted ({outcomeLabel})</h3>
          <ObservedVsPredictedChart data={monthlySummary} type="outcome" label={outcomeLabel} />
        </div>
        <div className="chart-card">
          <h3>Observed vs Predicted (Action)</h3>
          <ObservedVsPredictedChart data={monthlySummary} type="action" />
        </div>
      </div>

      <div className="chart-grid">
        <div className="chart-card">
          <h3>Incremental Lift by Month</h3>
          <IncrementalByMonthChart data={monthlySummary} outcomeLabel={outcomeLabel} />
        </div>
        <div className="chart-card">
          <h3>Feature Importance (Path A)</h3>
          {pathAFeatures ? (
            <FeatureImportanceChart data={pathAFeatures} color="#2563eb" />
          ) : (
            <p style={{ color: '#64748b', fontSize: 13 }}>No feature importance data available.</p>
          )}
        </div>
      </div>

      {pathBFeatures && (
        <div className="card" style={{ marginTop: 0 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>
            Feature Importance (Path B: Action &rarr; {outcomeLabel})
          </h3>
          <FeatureImportanceChart data={pathBFeatures} color="#16a34a" />
        </div>
      )}

      {/* ── Monthly Summary Table ──────────────────── */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-title">Monthly Aggregated Results</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Month</th>
              <th>Obs. Action</th>
              <th>Pred. Action</th>
              <th>Incr. Action</th>
              <th>Obs. {outcomeLabel}</th>
              <th>Pred. {outcomeLabel}</th>
              <th>Incr. {outcomeLabel}</th>
              <th>Lift %</th>
            </tr>
          </thead>
          <tbody>
            {monthlySummary.map((row) => (
              <tr key={row.month}>
                <td>{row.month}</td>
                <td>{row.totalObservedAction.toFixed(1)}</td>
                <td>{row.totalPredictedAction.toFixed(1)}</td>
                <td>{row.totalIncrementalAction.toFixed(2)}</td>
                <td>{row.totalObservedOutcome.toFixed(1)}</td>
                <td>{row.totalPredictedOutcome.toFixed(1)}</td>
                <td>{row.totalIncrementalOutcome.toFixed(2)}</td>
                <td>{row.liftPercentOutcome.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
