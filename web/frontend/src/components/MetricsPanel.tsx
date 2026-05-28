import type { MetricsReport } from '../types'

function fmt(v: number | undefined, decimals = 2): string {
  return (v ?? 0).toFixed(decimals)
}
function fmtPct(v: number | undefined): string {
  return ((v ?? 0) * 100).toFixed(1) + '%'
}
function colorScale(v: number, good: 'high' | 'low' = 'high'): string {
  if (good === 'high') return v >= 0.9 ? '#22c55e' : v >= 0.7 ? '#eab308' : '#ef4444'
  return v <= 0.05 ? '#22c55e' : v <= 0.15 ? '#eab308' : '#ef4444'
}

interface Props { metrics?: MetricsReport }

export default function MetricsPanel({ metrics }: Props) {
  if (!metrics) {
    return (
      <div style={{ background: '#111827', borderRadius: '10px', border: '1px solid #1e293b', padding: '16px' }}>
        <div style={{ color: '#4b5563', fontSize: '12px', textAlign: 'center' }}>Waiting for data...</div>
      </div>
    )
  }

  return (
    <div style={{ background: '#111827', borderRadius: '10px', border: '1px solid #1e293b', padding: '14px' }}>
      <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: '#38bdf8', letterSpacing: '0.5px' }}>
        DETECTION METRICS
      </div>

      {/* Confusion Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '14px' }}>
        {[
          { label: 'TP', value: metrics.tp, color: '#166534', text: '#86efac' },
          { label: 'FP', value: metrics.fp, color: '#7f1d1d', text: '#fca5a5' },
          { label: 'FN', value: metrics.fn, color: '#7f1d1d', text: '#fca5a5' },
          { label: 'TN', value: metrics.tn, color: '#166534', text: '#86efac' },
        ].map(m => (
          <div key={m.label} style={{
            background: m.color, padding: '8px', borderRadius: '6px', textAlign: 'center',
          }}>
            <div style={{ fontSize: '10px', color: m.text, fontWeight: 600 }}>{m.label}</div>
            <div style={{ fontSize: '20px', fontWeight: 800, color: '#fff' }}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* Rates */}
      {[
        { label: 'TPR (Detection)', value: fmtPct(metrics.detection_rate), color: colorScale(metrics.detection_rate) },
        { label: 'FPR', value: fmtPct(metrics.false_positive_rate), color: colorScale(metrics.false_positive_rate, 'low') },
        { label: 'Precision', value: fmtPct(metrics.precision), color: colorScale(metrics.precision) },
        { label: 'F1 Score', value: fmt(metrics.f1), color: colorScale(metrics.f1) },
      ].map(r => (
        <div key={r.label} style={{
          display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontSize: '12px',
        }}>
          <span style={{ color: '#6b7280' }}>{r.label}</span>
          <span style={{ fontWeight: 700, fontFamily: 'monospace', color: r.color }}>{r.value}</span>
        </div>
      ))}

      <div style={{ borderTop: '1px solid #1e293b', margin: '8px 0' }} />

      {[
        { label: 'MTTD', value: `${fmt(metrics.mttd_sec)}s` },
        { label: 'MTTI', value: `${fmt(metrics.mtti_sec)}s` },
        { label: 'Reward', value: fmt(metrics.cumulative_reward, 1),
          color: (metrics.cumulative_reward ?? 0) >= 0 ? '#22c55e' : '#ef4444' },
      ].map(r => (
        <div key={r.label} style={{
          display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontSize: '12px',
        }}>
          <span style={{ color: '#6b7280' }}>{r.label}</span>
          <span style={{ fontWeight: 700, fontFamily: 'monospace', color: r.color || '#e2e8f0' }}>{r.value}</span>
        </div>
      ))}
    </div>
  )
}
