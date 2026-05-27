const ACTIONS = [
  { id: 0, name: 'Allow', color: '#22c55e', gt: 'normal' },
  { id: 1, name: 'Flag', color: '#eab308', gt: 'attack' },
  { id: 2, name: 'Block', color: '#ef4444', gt: 'attack' },
  { id: 3, name: 'Isolate', color: '#a855f7', gt: 'attack' },
]

function sendAction(actionId: number, gt: string) {
  fetch(`/api/action?action=${actionId}&ground_truth=${gt}`, { method: 'POST' })
    .catch(() => {})
}

export default function ControlPanel() {
  return (
    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
      <span style={{ fontSize: '11px', color: '#64748b', marginRight: '4px' }}>Manual:</span>
      {ACTIONS.map(a => (
        <button
          key={a.id}
          onClick={() => sendAction(a.id, a.gt)}
          style={{
            background: a.color,
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            opacity: 0.85,
          }}
          onMouseEnter={e => { (e.target as HTMLElement).style.opacity = '1' }}
          onMouseLeave={e => { (e.target as HTMLElement).style.opacity = '0.85' }}
        >
          {a.name}
        </button>
      ))}
    </div>
  )
}
