import type { EventEntry } from '../types'

const ACTION_COLORS: Record<string, string> = {
  allow: '#22c55e', flag: '#eab308', block: '#ef4444', isolate: '#a855f7', info: '#38bdf8',
}

interface Props { events?: EventEntry[] }

export default function EventLog({ events }: Props) {
  const items = events || []

  return (
    <div style={{
      background: '#111827', borderRadius: '10px', border: '1px solid #1e293b',
      padding: '14px', flex: 1, overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '8px', color: '#38bdf8', letterSpacing: '0.5px' }}>
        EVENT LOG
      </div>
      <div style={{
        flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: '2px',
      }}>
        {items.length === 0 ? (
          <div style={{ color: '#374151', fontSize: '11px', textAlign: 'center', padding: '20px 0' }}>
            No events yet...
          </div>
        ) : (
          [...items].reverse().map((evt, i) => {
            const time = new Date(evt.timestamp * 1000)
            const timeStr = time.toLocaleTimeString()
            const color = ACTION_COLORS[evt.action_name] || '#4b5563'
            const msg = evt.message || ''
            const target = evt.target ? ` s${evt.target.dpid}:p${evt.target.port}` : ''
            const attackType = evt.attack_type && evt.attack_type !== 'none' ? ` ${evt.attack_type}` : ''
            const isAttack = msg.toLowerCase().includes('attack') || msg.toLowerCase().includes('spoof') || msg.toLowerCase().includes('rogue')
            return (
              <div key={i} style={{
                fontSize: '10px', fontFamily: "'JetBrains Mono', monospace",
                padding: '4px 8px',
                background: isAttack ? 'rgba(239,68,68,0.08)' : 'rgba(15,23,42,0.5)',
                borderRadius: '4px', borderLeft: `3px solid ${color}`,
              }}>
                <span style={{ color: '#4b5563' }}>{timeStr}</span>{' '}
                <span style={{ color, fontWeight: 700, fontSize: '9px' }}>{evt.action_name.toUpperCase()}</span>{' '}
                {attackType && <span style={{ color: '#fbbf24' }}>{attackType}</span>}{' '}
                {target && <span style={{ color: '#93c5fd' }}>{target}</span>}{' '}
                <span style={{ color: isAttack ? '#fca5a5' : '#9ca3af' }}>{msg}</span>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
