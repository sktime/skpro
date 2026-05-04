import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Scatter
} from 'recharts';

const API = 'http://localhost:8000';

// ── Utility ─────────────────────────────────────────────────────────────────

const fmt = (n, decimals = 3) =>
  typeof n === 'number' ? n.toFixed(decimals) : n;

// ── Sub-components ───────────────────────────────────────────────────────────

function Badge({ children, color = 'accent' }) {
  const colors = {
    accent: { bg: '#00d4ff22', border: '#00d4ff44', text: '#00d4ff' },
    purple: { bg: '#7c3aed22', border: '#7c3aed44', text: '#a78bfa' },
    green:  { bg: '#10b98122', border: '#10b98144', text: '#34d399' },
    red:    { bg: '#ef444422', border: '#ef444444', text: '#f87171' },
  };
  const c = colors[color] || colors.accent;
  return (
    <span style={{
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
      padding: '2px 10px', borderRadius: 20, fontSize: 11,
      fontFamily: 'var(--mono)', fontWeight: 700, letterSpacing: 1
    }}>
      {children}
    </span>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 24, ...style
    }}>
      {children}
    </div>
  );
}

function Btn({ children, onClick, disabled, variant = 'primary', small = false }) {
  const styles = {
    primary: { background: 'var(--accent)', color: '#0a0e1a' },
    secondary: { background: 'transparent', color: 'var(--accent)', border: '1px solid var(--accent)' },
    purple: { background: 'var(--accent2)', color: '#fff' },
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        ...styles[variant],
        padding: small ? '6px 14px' : '10px 22px',
        borderRadius: 8,
        border: styles[variant].border || 'none',
        fontWeight: 700,
        fontSize: small ? 12 : 14,
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'opacity 0.2s',
        fontFamily: 'var(--sans)',
      }}
    >
      {children}
    </button>
  );
}

function Spinner() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 32 }}>
      <div style={{
        width: 36, height: 36, border: '3px solid var(--border)',
        borderTop: '3px solid var(--accent)', borderRadius: '50%',
        animation: 'spin 0.8s linear infinite'
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function ErrorBox({ msg }) {
  if (!msg) return null;
  return (
    <div style={{
      background: '#ef444415', border: '1px solid #ef444440', borderRadius: 8,
      padding: '12px 16px', color: '#f87171', fontSize: 13, marginTop: 12
    }}>
      ⚠ {msg}
    </div>
  );
}

// ── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({ label, value, unit = '', color = 'var(--accent)' }) {
  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 10, padding: '14px 18px', flex: 1, minWidth: 120
    }}>
      <div style={{ color: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--mono)', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ color, fontSize: 22, fontWeight: 700, fontFamily: 'var(--mono)' }}>
        {fmt(value)}<span style={{ fontSize: 12, color: 'var(--text-muted)' }}> {unit}</span>
      </div>
    </div>
  );
}

// ── Prediction Chart ─────────────────────────────────────────────────────────

function PredChart({ predictions, modelName, color = '#00d4ff' }) {
  if (!predictions || predictions.length === 0) return null;

  const data = predictions.map((p, i) => ({
    idx: i,
    actual: parseFloat(p.y_true.toFixed(3)),
    predicted: parseFloat(p.y_pred.toFixed(3)),
    band: [parseFloat(p.y_lower.toFixed(3)), parseFloat(p.y_upper.toFixed(3))],
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '10px 14px', fontSize: 12
      }}>
        <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>Sample #{label}</div>
        {payload.map((p, i) => (
          <div key={i} style={{ color: p.color || p.stroke }}>{p.name}: {p.value}</div>
        ))}
      </div>
    );
  };

  return (
    <div>
      <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
        {modelName} — Predictions vs Actual with 80–90% Uncertainty Bands
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
          <XAxis dataKey="idx" stroke="#475569" tick={{ fontSize: 11 }} label={{ value: 'Test Sample Index', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 11 }} />
          <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />

          {/* Uncertainty band */}
          <Area
            dataKey="band"
            fill={color}
            fillOpacity={0.12}
            stroke="none"
            name="Uncertainty Band"
          />

          {/* Actual */}
          <Scatter dataKey="actual" fill="#f59e0b" name="Actual" r={3} />

          {/* Predicted */}
          <Line
            type="monotone"
            dataKey="predicted"
            stroke={color}
            strokeWidth={2}
            dot={false}
            name="Predicted"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Model Summary Panel ──────────────────────────────────────────────────────

function ModelSummary({ summary }) {
  if (!summary) return null;
  const r2Color = summary.r2_score > 0.8 ? 'var(--accent3)' : summary.r2_score > 0.5 ? 'var(--warning)' : 'var(--error)';

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <MetricCard label="R² Score" value={summary.r2_score} color={r2Color} />
        <MetricCard label="MAE" value={summary.mae} />
        <MetricCard label="RMSE" value={summary.rmse} />
        <MetricCard label="Train Samples" value={summary.n_train_samples} color="var(--accent2)" />
        <MetricCard label="Test Samples" value={summary.n_test_samples} color="var(--accent2)" />
        <MetricCard label="Training Time" value={summary.training_time_seconds} unit="s" color="var(--accent3)" />
      </div>

      <div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, fontFamily: 'var(--mono)' }}>
          MODEL PARAMETERS
        </div>
        <div style={{
          background: 'var(--surface2)', borderRadius: 8, padding: 16,
          fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 1.8
        }}>
          {Object.entries(summary.parameters).map(([k, v]) => (
            <div key={k}>
              <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
              <span style={{ color: 'var(--accent)' }}>{String(v)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Upload Tab ───────────────────────────────────────────────────────────────

function UploadTab({ onDatasetLoaded }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await axios.post(`${API}/upload`, form);
      onDatasetLoaded(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>
        📁 Upload Dataset
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files[0];
          if (f && f.name.endsWith('.csv')) setFile(f);
          else setError('Only CSV files are supported');
        }}
        style={{
          border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: 12, padding: '40px 20px', textAlign: 'center',
          background: dragOver ? '#00d4ff08' : 'var(--surface2)',
          transition: 'all 0.2s', cursor: 'pointer'
        }}
        onClick={() => document.getElementById('csv-input').click()}
      >
        <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>
          {file ? file.name : 'Drop CSV here or click to browse'}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          Supports CSV files with numeric columns
        </div>
        <input
          id="csv-input"
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={e => setFile(e.target.files[0])}
        />
      </div>

      <div style={{ marginTop: 16 }}>
        <Btn onClick={handleUpload} disabled={!file || loading}>
          {loading ? 'Uploading...' : 'Upload & Analyze'}
        </Btn>
      </div>

      <ErrorBox msg={error} />

      {/* Tip about sample datasets */}
      <div style={{
        marginTop: 20, padding: '12px 16px',
        background: '#00d4ff08', borderRadius: 8, border: '1px solid #00d4ff20',
        fontSize: 12, color: 'var(--text-muted)'
      }}>
        💡 <strong style={{ color: 'var(--text)' }}>Sample datasets</strong> are in the{' '}
        <code style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>sample_datasets/</code>{' '}
        folder: house_prices.csv, energy_consumption.csv, medical_data.csv, salary_data.csv
      </div>
    </Card>
  );
}

// ── Dataset Info Panel ───────────────────────────────────────────────────────

function DatasetInfo({ dataset }) {
  if (!dataset) return null;
  return (
    <Card style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{ fontWeight: 700, fontSize: 16 }}>✅ Dataset Loaded</div>
        <Badge color="green">{dataset.shape.rows} rows × {dataset.shape.columns} cols</Badge>
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, fontFamily: 'var(--mono)' }}>DATASET ID</div>
          <code style={{ color: 'var(--accent)', fontFamily: 'var(--mono)', fontSize: 13 }}>{dataset.dataset_id}</code>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, fontFamily: 'var(--mono)' }}>NUMERIC COLUMNS</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {dataset.numeric_columns.map(c => (
              <Badge key={c} color="purple">{c}</Badge>
            ))}
          </div>
        </div>
      </div>

      {/* Preview table */}
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, fontFamily: 'var(--mono)' }}>
        PREVIEW (first 5 rows)
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
          <thead>
            <tr>
              {dataset.all_columns.map(c => (
                <th key={c} style={{
                  textAlign: 'left', padding: '6px 12px',
                  color: 'var(--text-muted)', fontFamily: 'var(--mono)',
                  borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap'
                }}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataset.preview.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'var(--surface2)' }}>
                {dataset.all_columns.map(c => (
                  <td key={c} style={{ padding: '5px 12px', borderBottom: '1px solid var(--border)', fontFamily: 'var(--mono)', color: 'var(--accent)' }}>
                    {typeof row[c] === 'number' ? row[c].toFixed(2) : row[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ── Predict Tab ──────────────────────────────────────────────────────────────

function PredictTab({ dataset, models }) {
  const [target, setTarget] = useState('');
  const [features, setFeatures] = useState([]);
  const [modelType, setModelType] = useState('bayesian_ridge');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const numCols = dataset?.numeric_columns || [];

  const toggleFeature = col => {
    setFeatures(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  const handlePredict = async () => {
    if (!target || features.length === 0) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await axios.post(`${API}/predict`, {
        dataset_id: dataset.dataset_id,
        target_column: target,
        feature_columns: features,
        model_type: modelType,
        test_size: 0.2,
      });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  if (!dataset) return (
    <Card>
      <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>
        ← Upload a dataset first to run predictions
      </div>
    </Card>
  );

  return (
    <div>
      <Card>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>⚙️ Configure Prediction</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          {/* Target */}
          <div>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 8 }}>
              TARGET COLUMN (what to predict)
            </label>
            <select
              value={target}
              onChange={e => { setTarget(e.target.value); setFeatures([]); }}
              style={{
                width: '100%', background: 'var(--surface2)', border: '1px solid var(--border)',
                color: 'var(--text)', borderRadius: 8, padding: '10px 12px', fontSize: 14
              }}
            >
              <option value="">— Select target —</option>
              {numCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          {/* Model */}
          <div>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 8 }}>
              PROBABILISTIC MODEL
            </label>
            <select
              value={modelType}
              onChange={e => setModelType(e.target.value)}
              style={{
                width: '100%', background: 'var(--surface2)', border: '1px solid var(--border)',
                color: 'var(--text)', borderRadius: 8, padding: '10px 12px', fontSize: 14
              }}
            >
              {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
            {models.find(m => m.id === modelType) && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                {models.find(m => m.id === modelType)?.description}
              </div>
            )}
          </div>
        </div>

        {/* Feature columns */}
        {target && (
          <div style={{ marginTop: 20 }}>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 10 }}>
              FEATURE COLUMNS (input variables)
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {numCols.filter(c => c !== target).map(c => (
                <button
                  key={c}
                  onClick={() => toggleFeature(c)}
                  style={{
                    padding: '6px 14px', borderRadius: 20, border: '1px solid',
                    borderColor: features.includes(c) ? 'var(--accent)' : 'var(--border)',
                    background: features.includes(c) ? '#00d4ff18' : 'transparent',
                    color: features.includes(c) ? 'var(--accent)' : 'var(--text-muted)',
                    fontSize: 12, fontFamily: 'var(--mono)', cursor: 'pointer', transition: 'all 0.15s'
                  }}
                >
                  {c}
                </button>
              ))}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              {features.length} feature{features.length !== 1 ? 's' : ''} selected
            </div>
          </div>
        )}

        <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 16 }}>
          <Btn onClick={handlePredict} disabled={!target || features.length === 0 || loading}>
            {loading ? '⏳ Running...' : '🔮 Run Prediction'}
          </Btn>
          {loading && <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>Fitting model…</span>}
        </div>

        <ErrorBox msg={error} />
      </Card>

      {/* Results */}
      {result && (
        <Card style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, fontSize: 16 }}>📈 Results</div>
            <Badge>{models.find(m => m.id === modelType)?.name}</Badge>
          </div>
          <PredChart predictions={result.predictions} modelName={models.find(m => m.id === modelType)?.name} />
          <div style={{ height: 1, background: 'var(--border)', margin: '24px 0' }} />
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>📊 Model Summary</div>
          <ModelSummary summary={result.model_summary} />
        </Card>
      )}
    </div>
  );
}

// ── Compare Tab ──────────────────────────────────────────────────────────────

function CompareTab({ dataset, models }) {
  const [target, setTarget] = useState('');
  const [features, setFeatures] = useState([]);
  const [selectedModels, setSelectedModels] = useState(['bayesian_ridge', 'gaussian_process']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const numCols = dataset?.numeric_columns || [];

  const toggleModel = id => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  const toggleFeature = col => {
    setFeatures(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  const handleCompare = async () => {
    if (!target || features.length === 0 || selectedModels.length < 2) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await axios.post(`${API}/compare`, {
        dataset_id: dataset.dataset_id,
        target_column: target,
        feature_columns: features,
        model_types: selectedModels,
        test_size: 0.2,
      });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  if (!dataset) return (
    <Card>
      <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>
        ← Upload a dataset first
      </div>
    </Card>
  );

  const COLORS = ['#00d4ff', '#a78bfa', '#34d399', '#f59e0b'];

  return (
    <div>
      <Card>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>⚖️ Model Comparison</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <div>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 8 }}>
              TARGET COLUMN
            </label>
            <select
              value={target}
              onChange={e => { setTarget(e.target.value); setFeatures([]); }}
              style={{
                width: '100%', background: 'var(--surface2)', border: '1px solid var(--border)',
                color: 'var(--text)', borderRadius: 8, padding: '10px 12px', fontSize: 14
              }}
            >
              <option value="">— Select target —</option>
              {numCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 8 }}>
              MODELS TO COMPARE (select ≥ 2)
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {models.map((m, i) => (
                <button
                  key={m.id}
                  onClick={() => toggleModel(m.id)}
                  style={{
                    padding: '6px 14px', borderRadius: 20, border: '1px solid',
                    borderColor: selectedModels.includes(m.id) ? COLORS[i % COLORS.length] : 'var(--border)',
                    background: selectedModels.includes(m.id) ? `${COLORS[i % COLORS.length]}18` : 'transparent',
                    color: selectedModels.includes(m.id) ? COLORS[i % COLORS.length] : 'var(--text-muted)',
                    fontSize: 12, cursor: 'pointer', transition: 'all 0.15s', fontFamily: 'var(--sans)'
                  }}
                >
                  {m.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {target && (
          <div style={{ marginTop: 20 }}>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 10 }}>
              FEATURE COLUMNS
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {numCols.filter(c => c !== target).map(c => (
                <button
                  key={c}
                  onClick={() => toggleFeature(c)}
                  style={{
                    padding: '6px 14px', borderRadius: 20, border: '1px solid',
                    borderColor: features.includes(c) ? 'var(--accent)' : 'var(--border)',
                    background: features.includes(c) ? '#00d4ff18' : 'transparent',
                    color: features.includes(c) ? 'var(--accent)' : 'var(--text-muted)',
                    fontSize: 12, fontFamily: 'var(--mono)', cursor: 'pointer'
                  }}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
        )}

        <div style={{ marginTop: 20 }}>
          <Btn onClick={handleCompare} disabled={!target || features.length === 0 || selectedModels.length < 2 || loading}>
            {loading ? '⏳ Comparing...' : '⚖️ Compare Models'}
          </Btn>
        </div>
        <ErrorBox msg={error} />
      </Card>

      {result && (
        <div>
          {/* Metrics comparison table */}
          <Card style={{ marginTop: 16 }}>
            <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 16 }}>📊 Comparison Summary</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Model', 'R² Score', 'MAE', 'RMSE', 'Time (s)'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-muted)', fontFamily: 'var(--mono)', fontSize: 11 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.comparison.map((r, i) => {
                  const bestR2 = Math.max(...result.comparison.filter(x => x.summary).map(x => x.summary.r2_score));
                  const isBest = r.summary?.r2_score === bestR2;
                  return (
                    <tr key={r.model_type} style={{ borderBottom: '1px solid var(--border)', background: isBest ? '#00d4ff06' : 'transparent' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, color: COLORS[i % COLORS.length] }}>
                        {models.find(m => m.id === r.model_type)?.name}
                        {isBest && <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--accent3)' }}>★ BEST</span>}
                      </td>
                      {r.error ? (
                        <td colSpan={4} style={{ padding: '10px 12px', color: 'var(--error)', fontSize: 12 }}>Error: {r.error}</td>
                      ) : (
                        <>
                          <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)', color: r.summary.r2_score > 0.8 ? 'var(--accent3)' : 'var(--text)' }}>{fmt(r.summary.r2_score)}</td>
                          <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)' }}>{fmt(r.summary.mae)}</td>
                          <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)' }}>{fmt(r.summary.rmse)}</td>
                          <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)' }}>{fmt(r.summary.training_time_seconds)}</td>
                        </>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          {/* Per-model charts */}
          {result.comparison.map((r, i) => !r.error && (
            <Card key={r.model_type} style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: COLORS[i % COLORS.length] }} />
                <div style={{ fontWeight: 700 }}>{models.find(m => m.id === r.model_type)?.name}</div>
                <Badge color={i === 0 ? 'accent' : 'purple'}>R² = {fmt(r.summary.r2_score)}</Badge>
              </div>
              <PredChart predictions={r.predictions} modelName={models.find(m => m.id === r.model_type)?.name} color={COLORS[i % COLORS.length]} />
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Models Tab ───────────────────────────────────────────────────────────────

function ModelsTab({ models }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
      {models.map((m, i) => {
        const colors = ['var(--accent)', 'var(--accent2)', 'var(--accent3)', 'var(--warning)'];
        const c = colors[i % colors.length];
        return (
          <Card key={m.id} style={{ borderTop: `3px solid ${c}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 15, color: c }}>{m.name}</div>
              <Badge color={['accent', 'purple', 'green', 'accent'][i % 4]}>{m.category}</Badge>
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 12 }}>{m.description}</div>
            <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-muted)', marginBottom: 4 }}>ASSUMPTIONS</div>
            <div style={{ fontSize: 12, color: 'var(--text)', marginBottom: 10 }}>{m.assumptions}</div>
            <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-muted)', marginBottom: 4 }}>STRENGTHS</div>
            <div style={{ fontSize: 12, color: 'var(--accent3)' }}>{m.strengths}</div>
            <div style={{ marginTop: 12 }}>
              <code style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-muted)', background: 'var(--surface2)', padding: '2px 8px', borderRadius: 4 }}>
                id: {m.id}
              </code>
            </div>
          </Card>
        );
      })}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState('upload');
  const [dataset, setDataset] = useState(null);
  const [models, setModels] = useState([]);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    axios.get(`${API}/models`)
      .then(res => { setModels(res.data.models); setApiStatus('online'); })
      .catch(() => setApiStatus('offline'));
  }, []);

  const TABS = [
    { id: 'upload', label: '📁 Upload', desc: 'Load Dataset' },
    { id: 'predict', label: '🔮 Predict', desc: 'Run Model' },
    { id: 'compare', label: '⚖️ Compare', desc: 'Side by Side' },
    { id: 'models', label: '📖 Models', desc: 'Documentation' },
  ];

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      {/* Header */}
      <div style={{
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface)',
        padding: '0 32px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 60, position: 'sticky', top: 0, zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ fontFamily: 'var(--mono)', fontWeight: 700, fontSize: 18, color: 'var(--accent)' }}>
            sk<span style={{ color: 'var(--accent2)' }}>pro</span>
          </div>
          <div style={{ width: 1, height: 24, background: 'var(--border)' }} />
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            Probabilistic ML Dashboard
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {dataset && <Badge color="green">Dataset: {dataset.dataset_id}</Badge>}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: apiStatus === 'online' ? 'var(--accent3)' : apiStatus === 'offline' ? 'var(--error)' : 'var(--warning)',
              boxShadow: apiStatus === 'online' ? '0 0 8px var(--accent3)' : 'none'
            }} />
            <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
              API {apiStatus}
            </span>
          </div>
        </div>
      </div>

      {/* Nav */}
      <div style={{
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface)',
        padding: '0 32px',
        display: 'flex', gap: 4
      }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '14px 20px',
              background: 'transparent',
              border: 'none',
              borderBottom: `2px solid ${tab === t.id ? 'var(--accent)' : 'transparent'}`,
              color: tab === t.id ? 'var(--accent)' : 'var(--text-muted)',
              fontWeight: tab === t.id ? 700 : 400,
              fontSize: 13,
              cursor: 'pointer',
              transition: 'all 0.15s',
              display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2
            }}
          >
            <span>{t.label}</span>
            <span style={{ fontSize: 10, opacity: 0.6 }}>{t.desc}</span>
          </button>
        ))}
      </div>

      {/* API offline warning */}
      {apiStatus === 'offline' && (
        <div style={{
          background: '#ef444415', border: '1px solid #ef444440',
          padding: '10px 32px', fontSize: 13, color: '#f87171'
        }}>
          ⚠ Backend API is offline. Start it with:{' '}
          <code style={{ fontFamily: 'var(--mono)' }}>cd backend && uvicorn main:app --reload</code>
        </div>
      )}

      {/* Content */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px' }}>
        {tab === 'upload' && (
          <div>
            <UploadTab onDatasetLoaded={d => { setDataset(d); setTab('predict'); }} />
            <DatasetInfo dataset={dataset} />
          </div>
        )}
        {tab === 'predict' && <PredictTab dataset={dataset} models={models} />}
        {tab === 'compare' && <CompareTab dataset={dataset} models={models} />}
        {tab === 'models' && <ModelsTab models={models} />}
      </div>
    </div>
  );
}
