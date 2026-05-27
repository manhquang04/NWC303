import type { MetricsReport } from '../types'

const cardStyle: React.CSSProperties = {
  background: '#1e293b',
  borderRadius: '8px',
  border: '1px solid #334155',
  padding: '16px',
}

const metricRow: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  padding: '4px 0',
  fontSize: '13px',
}

const metricLabel: React.CSSProperties = {
  color: '#94a3b8',
}

const metricValue = (val: number, fmt: string = '.2f'): React.CSSProperties => ({
  fontWeight: 600,
  fontFamily: 'monospace',
})

function fmt(v: number, decimals = 2): string {
  return v.toFixed(decimals)
}

function fmtPct(v: number): string {
  return (v * 100).toFixed(1) + '%'
}

function colorScale(v: number, good: 'high' | 'low' = 'high'): string {
  if (good === 'high') {
    if (v >= 0.9) return '#22c55e'
    if (v >= 0.7) return '#eab308'
    return '#ef4444'
  } else {
    if (v <= 0.05) return '#22c55e'
    if (v <= 0.15) return '#eab308'
    return '#ef4444'
  }
}

interface Props {
  metrics?: MetricsReport
}

export default function MetricsPanel({ metrics }: Props) {
  if (!metrics) {
    return <div style={cardStyle}><div style={{ color: '#64748b', fontSize: '13px' }}>Waiting for data...</div></div>
  }

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#38bdf8' }}>
        Detection Metrics
      </div>

      {/* Confusion Matrix Mini */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '4px',
        marginBottom: '12px',
      }}>
        <div style={{ background: '#166534', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#86efac' }}>TP</div>
          <div style={{ fontSize: '18px', fontWeight: 700 }}>{metrics.tp}</div>
        </div>
        <div style={{ background: '#991b1b', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#fca5a5' }}>FP</div>
          <div style={{ fontSize: '18px', fontWeight: 700 }}>{metrics.fp}</div>
        </div>
        <div style={{ background: '#991b1b', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#fca5a5' }}>FN</div>
          <div style={{ fontSize: '18px', fontWeight: 700 }}>{metrics.fn}</div>
        </div>
        <div style={{ background: '#166534', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#86efac' }}>TN</div>
          <div style={{ fontSize: '18px', fontWeight: 700 }}>{metrics.tn}</div>
        </div>
      </div>

      {/* Rates */}
      <div style={metricRow}>
        <span style={metricLabel}>TPR (Detection)</span>
        <span style={{ ...metricValue(metrics.detection_rate), color: colorScale(metrics.detection_rate) }}>
          {fmtPct(metrics.detection_rate)}
        </span>
      </div>
      <div style={metricRow}>
        <span style={metricLabel}>FPR</span>
        <span style={{ ...metricValue(metrics.false_positive_rate), color: colorScale(metrics.false_positive_rate, 'low') }}>
          {fmtPct(metrics.false_positive_rate)}
        </span>
      </div>
      <div style={metricRow}>
        <span style={metricLabel}>Precision</span>
        <span style={{ ...metricValue(metrics.precision), color: colorScale(metrics.precision) }}>
          {fmtPct(metrics.precision)}
        </span>
      </div>
      <div style={metricRow}>
        <span style={metricLabel}>F1 Score</span>
        <span style={{ ...metricValue(metrics.f1), color: colorScale(metrics.f1) }}>
          {fmt(metrics.f1)}
        </span>
      </div>

      <div style={{ borderTop: '1px solid #334155', margin: '8px 0' }} />

      <div style={metricRow}>
        <span style={metricLabel}>MTTD</span>
        <span style={{ fontWeight: 600, fontFamily: 'monospace' }}>{fmt(metrics.mttd_sec)}s</span>
      </div>
      <div style={metricRow}>
        <span style={metricLabel}>MTTI</span>
        <span style={{ fontWeight: 600, fontFamily: 'monospace' }}>{fmt(metrics.mtti_sec)}s</span>
      </div>
      <div style={metricRow}>
        <span style={metricLabel}>Reward</span>
        <span style={{ fontWeight: 600, fontFamily: 'monospace', color: metrics.cumulative_reward >= 0 ? '#22c55e' : '#ef4444' }}>
          {fmt(metrics.cumulative_reward, 1)}
        </span>
      </div>
    </div>
  )
}
