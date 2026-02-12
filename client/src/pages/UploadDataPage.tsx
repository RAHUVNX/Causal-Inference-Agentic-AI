import { useState, useRef } from 'react';
import { uploadData, UploadResponse } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

export default function UploadDataPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && dropped.name.endsWith('.csv')) {
      setFile(dropped);
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await uploadData(file);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h2>Upload Data</h2>
        <p>Upload a CSV file containing HCP observation data for causal lift analysis.</p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">CSV File Upload</div>

        <div
          className={`file-upload-zone ${file ? 'has-file' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          {file ? (
            <>
              <p>Selected file:</p>
              <p className="file-name">{file.name}</p>
              <p>({(file.size / 1024).toFixed(1)} KB)</p>
            </>
          ) : (
            <>
              <p>Click or drag a CSV file here to upload</p>
              <p style={{ marginTop: 8, fontSize: 12 }}>
                Required columns: hcp_id, month, suggestion_type, action_type,
                outcome_type, suggestion_count, action_count, outcome_count
              </p>
            </>
          )}
        </div>

        <div style={{ marginTop: 16 }}>
          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading ? 'Uploading...' : 'Upload Data'}
          </button>
        </div>
      </div>

      {loading && <LoadingSpinner message="Parsing and storing observations..." />}

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="card">
          <div className="card-title">Upload Complete</div>
          <div className="alert alert-success">
            Successfully inserted {result.rowsInserted} observations (batch: {result.uploadBatchId})
          </div>

          <div style={{ marginTop: 16 }}>
            <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
              Distinct Values Found
            </h4>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Values</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Suggestion Types</td>
                  <td>{result.distinctValues.suggestionTypes.join(', ')}</td>
                </tr>
                <tr>
                  <td>Action Types</td>
                  <td>{result.distinctValues.actionTypes.join(', ')}</td>
                </tr>
                <tr>
                  <td>Outcome Types</td>
                  <td>{result.distinctValues.outcomeTypes.join(', ')}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
