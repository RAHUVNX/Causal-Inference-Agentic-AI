import { useState } from 'react';
import { runLift, LiftRequest, LiftResponse } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import KpiCard from '../components/KpiCard';
import IncrementalByMonthChart from '../charts/IncrementalByMonthChart';

export default function RunLiftPage() {
  const [form, setForm] = useState<LiftRequest>({
    suggestionType: '',
    actionType: '',
    outcomeType: 'trx',
    adstockEnabled: false,
    adstockDecay: 0.5,
    theme: '',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LiftResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: LiftRequest = {
        ...form,
        theme: form.theme || undefined,
      };
      const response = await runLift(payload);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  const outcomeLabel = form.outcomeType === 'trx' ? 'TRX' : 'NBRX';

  return (
    <div>
      <div className="page-header">
        <h2>Run Lift Simulation</h2>
        <p>
          Execute the full counterfactual simulation: train models, zero out suggestions,
          and compute incremental lift.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Simulation Configuration</div>
        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-group">
              <label>Suggestion Type</label>
              <input
                type="text"
                value={form.suggestionType}
                onChange={(e) => setForm({ ...form, suggestionType: e.target.value })}
                placeholder="e.g., email, call, sample"
                required
              />
            </div>
            <div className="form-group">
              <label>Action Type</label>
              <input
                type="text"
                value={form.actionType}
                onChange={(e) => setForm({ ...form, actionType: e.target.value })}
                placeholder="e.g., detail, sample_drop"
                required
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Outcome Type</label>
              <select
                value={form.outcomeType}
                onChange={(e) =>
                  setForm({ ...form, outcomeType: e.target.value as 'trx' | 'nbrx' })
                }
              >
                <option value="trx">TRX (Total Prescriptions)</option>
                <option value="nbrx">NBRX (New-to-Brand Prescriptions)</option>
              </select>
            </div>
            <div className="form-group">
              <label>Theme (Optional)</label>
              <input
                type="text"
                value={form.theme || ''}
                onChange={(e) => setForm({ ...form, theme: e.target.value })}
                placeholder="e.g., efficacy, safety"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="checkbox-group">
                <input
                  type="checkbox"
                  checked={form.adstockEnabled}
                  onChange={(e) => setForm({ ...form, adstockEnabled: e.target.checked })}
                />
                Enable Adstock Transformation
              </label>
            </div>
            {form.adstockEnabled && (
              <div className="form-group">
                <label>Adstock Decay Rate (0-1)</label>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={form.adstockDecay}
                  onChange={(e) =>
                    setForm({ ...form, adstockDecay: parseFloat(e.target.value) || 0.5 })
                  }
                />
              </div>
            )}
          </div>

          <button className="btn btn-success" type="submit" disabled={loading}>
            {loading ? 'Running Simulation...' : 'Run Full Counterfactual Simulation'}
          </button>
        </form>
      </div>

      {loading && (
        <LoadingSpinner message="Running counterfactual simulation pipeline: feature engineering → model training → zero-out → incremental lift..." />
      )}

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <>
          <div className="kpi-grid">
            <KpiCard
              label={`Total Incremental ${outcomeLabel}`}
              value={result.totalIncrementalOutcome}
              variant={result.totalIncrementalOutcome > 0 ? 'positive' : 'negative'}
              subtitle="Predicted - Counterfactual"
            />
            <KpiCard
              label="Total Incremental Action"
              value={result.totalIncrementalAction}
              variant={result.totalIncrementalAction > 0 ? 'positive' : 'negative'}
              subtitle="Predicted - Counterfactual"
            />
            <KpiCard
              label="Overall Lift %"
              value={`${result.overallLiftPercent.toFixed(2)}%`}
              variant={result.overallLiftPercent > 0 ? 'positive' : 'negative'}
              subtitle="Incremental / Counterfactual"
            />
            <KpiCard
              label="Observations"
              value={result.totalObservations}
              subtitle={`Run: ${result.runId}`}
            />
          </div>

          <div className="chart-grid">
            <div className="chart-card">
              <h3>Incremental Lift by Month</h3>
              <IncrementalByMonthChart
                data={result.monthlySummary}
                outcomeLabel={outcomeLabel}
              />
            </div>

            <div className="chart-card">
              <h3>Model Metrics</h3>
              <div className="metrics-grid">
                <div>
                  <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Path A</h4>
                  <div className="metric-item">
                    <span className="metric-label">R&sup2;</span>
                    <span className="metric-value">{result.metrics.pathA.rSquared.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">MAE</span>
                    <span className="metric-value">{result.metrics.pathA.mae.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">
                      {(result.metrics.pathA.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">F1</span>
                    <span className="metric-value">{result.metrics.pathA.f1.toFixed(4)}</span>
                  </div>
                </div>

                <div>
                  <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Path B</h4>
                  <div className="metric-item">
                    <span className="metric-label">R&sup2;</span>
                    <span className="metric-value">{result.metrics.pathB.rSquared.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">MAE</span>
                    <span className="metric-value">{result.metrics.pathB.mae.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">
                      {(result.metrics.pathB.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">F1</span>
                    <span className="metric-value">{result.metrics.pathB.f1.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Monthly Summary</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Month</th>
                  <th>Incr. Action</th>
                  <th>Incr. {outcomeLabel}</th>
                  <th>Lift % (Action)</th>
                  <th>Lift % ({outcomeLabel})</th>
                </tr>
              </thead>
              <tbody>
                {result.monthlySummary.map((row) => (
                  <tr key={row.month}>
                    <td>{row.month}</td>
                    <td>{row.totalIncrementalAction.toFixed(2)}</td>
                    <td>{row.totalIncrementalOutcome.toFixed(2)}</td>
                    <td>{row.liftPercentAction.toFixed(2)}%</td>
                    <td>{row.liftPercentOutcome.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
