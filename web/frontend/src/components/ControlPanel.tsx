export default function ControlPanel() {
  return (
    <div style={{
      display: 'flex',
      gap: '8px',
      alignItems: 'center',
    }}>
      <span style={{
        fontSize: '11px',
        color: '#64748b',
        padding: '4px 8px',
        background: '#0f172a',
        borderRadius: '4px',
        border: '1px solid #334155',
      }}>
        Actions: <span style={{ color: '#22c55e' }}>allow</span> · <span style={{ color: '#eab308' }}>flag</span> · <span style={{ color: '#ef4444' }}>block</span> · <span style={{ color: '#a855f7' }}>isolate</span>
      </span>
    </div>
  )
}
