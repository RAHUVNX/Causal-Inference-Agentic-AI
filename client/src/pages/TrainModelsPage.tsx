import { useState } from 'react';
import { trainModels, TrainRequest, TrainResponse } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

export default function TrainModelsPage() {
  const [form, setForm] = useState<TrainRequest>({
    suggestionType: '',
    actionType: '',
    outcomeType: 'trx',
    adstockEnabled: false,
    adstockDecay: 0.5,
    theme: '',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: TrainRequest = {
        ...form,
        theme: form.theme || undefined,
      };
      const response = await trainModels(payload);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Training failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h2>Train Models</h2>
        <p>
          Train the two-path causal models: Path A (Suggestion &rarr; Action) and
          Path B (Action &rarr; Outcome).
        </p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Training Configuration</div>
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

          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? 'Training...' : 'Train Path A & Path B Models'}
          </button>
        </form>
      </div>

      {loading && (
        <LoadingSpinner message="Training Path A (Suggestion → Action) and Path B (Action → Outcome) models..." />
      )}

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="card">
          <div className="card-title">Training Results</div>
          <div className="alert alert-success">
            Models trained successfully on {result.totalObservations} observations (Run ID: {result.runId})
          </div>

          <div className="metrics-grid" style={{ marginTop: 16 }}>
            <div>
              <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
                Path A: Suggestion &rarr; Action
              </h4>
              <div className="metric-item">
                <span className="metric-label">Model</span>
                <span className="metric-value">{result.pathA.modelName}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">R&sup2;</span>
                <span className="metric-value">{result.pathA.rSquared.toFixed(4)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">MAE</span>
                <span className="metric-value">{result.pathA.mae.toFixed(4)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Features</span>
                <span className="metric-value">{result.pathA.featureCount}</span>
              </div>
            </div>

            <div>
              <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
                Path B: Action &rarr; Outcome
              </h4>
              <div className="metric-item">
                <span className="metric-label">Model</span>
                <span className="metric-value">{result.pathB.modelName}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">R&sup2;</span>
                <span className="metric-value">{result.pathB.rSquared.toFixed(4)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">MAE</span>
                <span className="metric-value">{result.pathB.mae.toFixed(4)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Features</span>
                <span className="metric-value">{result.pathB.featureCount}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
