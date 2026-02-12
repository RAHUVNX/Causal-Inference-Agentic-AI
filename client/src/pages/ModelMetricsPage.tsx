import { useState, useEffect } from 'react';
import {
  getSummary,
  getFeatureImportance,
  SummaryResponse,
  FeatureImportanceResponse,
} from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import FeatureImportanceChart from '../charts/FeatureImportanceChart';

export default function ModelMetricsPage() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [featureData, setFeatureData] = useState<FeatureImportanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
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
        if (err.response?.status === 404) {
          setError('No model metrics available. Run a lift simulation first.');
        } else {
          setError(err.response?.data?.error || err.message || 'Failed to load metrics');
        }
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <LoadingSpinner message="Loading model metrics..." />;

  if (error) {
    return (
      <div>
        <div className="page-header">
          <h2>Model Metrics</h2>
          <p>Detailed performance metrics and feature importance for trained models.</p>
        </div>
        <div className="alert alert-warning">{error}</div>
      </div>
    );
  }

  if (!summary || !featureData) return null;

  const { modelRun } = summary;
  const outcomeLabel = summary.outcomeType === 'nbrx' ? 'NBRX' : 'TRX';

  return (
    <div>
      <div className="page-header">
        <h2>Model Metrics</h2>
        <p>
          Performance evaluation for run <strong>{summary.runId}</strong>
        </p>
      </div>

      {/* ── Model Run Metadata ─────────────────────── */}
      {modelRun && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title">Run Summary</div>
          <div className="form-row">
            <div>
              <div className="metric-item">
                <span className="metric-label">Status</span>
                <span className="metric-value">{modelRun.status}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Started</span>
                <span className="metric-value">
                  {new Date(modelRun.startedAt).toLocaleString()}
                </span>
              </div>
              {modelRun.completedAt && (
                <div className="metric-item">
                  <span className="metric-label">Completed</span>
                  <span className="metric-value">
                    {new Date(modelRun.completedAt).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
            <div>
              <div className="metric-item">
                <span className="metric-label">Observations</span>
                <span className="metric-value">{summary.totalObservations}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Outcome Type</span>
                <span className="metric-value">{outcomeLabel}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Regression Metrics ─────────────────────── */}
      <div className="metrics-grid">
        <div className="card">
          <div className="card-title">Path A: Suggestion &rarr; Action</div>
          <div className="metric-item">
            <span className="metric-label">R&sup2; (Coefficient of Determination)</span>
            <span className="metric-value">{modelRun?.rSquaredPathA?.toFixed(6) || 'N/A'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">MAE (Mean Absolute Error)</span>
            <span className="metric-value">{modelRun?.maePathA?.toFixed(6) || 'N/A'}</span>
          </div>
          <p style={{ fontSize: 12, color: '#64748b', marginTop: 12 }}>
            Path A predicts action_count from features including suggestion_count,
            percent_other_hcps_suggested, tenure, prior prescriptions, and demographics.
          </p>
        </div>

        <div className="card">
          <div className="card-title">Path B: Action &rarr; {outcomeLabel}</div>
          <div className="metric-item">
            <span className="metric-label">R&sup2; (Coefficient of Determination)</span>
            <span className="metric-value">{modelRun?.rSquaredPathB?.toFixed(6) || 'N/A'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">MAE (Mean Absolute Error)</span>
            <span className="metric-value">{modelRun?.maePathB?.toFixed(6) || 'N/A'}</span>
          </div>
          <p style={{ fontSize: 12, color: '#64748b', marginTop: 12 }}>
            Path B predicts {outcomeLabel.toLowerCase()} from features including action_count
            (which is substituted with predicted/counterfactual values during simulation).
          </p>
        </div>
      </div>

      {/* ── Feature Importance ─────────────────────── */}
      {featureData.models.map((model) => (
        <div className="card" key={model.modelName} style={{ marginTop: 24 }}>
          <div className="card-title">
            Feature Importance: {model.modelName}
          </div>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
            Features ranked by absolute coefficient magnitude. Higher values indicate
            greater influence on the model prediction.
          </p>
          <FeatureImportanceChart
            data={model.featureImportance}
            color={model.modelName.includes('PathA') ? '#2563eb' : '#16a34a'}
          />

          <div style={{ marginTop: 24 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Feature</th>
                  <th>Importance</th>
                </tr>
              </thead>
              <tbody>
                {model.featureImportance.map((fi) => (
                  <tr key={fi.featureName}>
                    <td>{fi.rank}</td>
                    <td>{fi.featureName}</td>
                    <td>{fi.importance.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
