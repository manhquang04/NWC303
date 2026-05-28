const ACTIONS = [
  { id: 0, name: 'Allow', color: '#22c55e', gt: 'normal', icon: '✓' },
  { id: 1, name: 'Flag', color: '#eab308', gt: 'attack', icon: '⚠' },
  { id: 2, name: 'Block', color: '#ef4444', gt: 'attack', icon: '✕' },
  { id: 3, name: 'Isolate', color: '#a855f7', gt: 'attack', icon: '⊘' },
]

function sendAction(actionId: number, gt: string) {
  fetch(`/api/action?action=${actionId}&ground_truth=${gt}`, { method: 'POST' }).catch(() => {})
}

export default function ControlPanel() {
  return (
    <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
      <span style={{ fontSize: '10px', color: '#4b5563', marginRight: '4px', fontWeight: 600 }}>TEST:</span>
      {ACTIONS.map(a => (
        <button
          key={a.id}
          onClick={() => sendAction(a.id, a.gt)}
          style={{
            background: `${a.color}22`, color: a.color,
            border: `1px solid ${a.color}44`, borderRadius: '6px',
            padding: '4px 12px', fontSize: '11px', fontWeight: 700,
            cursor: 'pointer', transition: 'all 0.2s',
            display: 'flex', alignItems: 'center', gap: '4px',
          }}
          onMouseEnter={e => {
            const el = e.target as HTMLElement
            el.style.background = `${a.color}44`
            el.style.boxShadow = `0 0 12px ${a.color}44`
          }}
          onMouseLeave={e => {
            const el = e.target as HTMLElement
            el.style.background = `${a.color}22`
            el.style.boxShadow = 'none'
          }}
        >
          {a.name}
        </button>
      ))}
    </div>
  )
}
