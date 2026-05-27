import type { EventEntry } from '../types'

const ACTION_COLORS: Record<string, string> = {
  allow: '#22c55e',
  flag: '#eab308',
  block: '#ef4444',
  isolate: '#a855f7',
  info: '#38bdf8',
}

interface Props {
  events?: EventEntry[]
}

export default function EventLog({ events }: Props) {
  const items = events || []

  return (
    <div style={{
      background: '#1e293b',
      borderRadius: '8px',
      border: '1px solid #334155',
      padding: '16px',
      flex: 1,
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: '#38bdf8' }}>
        Event Log
      </div>
      <div style={{
        flex: 1,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '3px',
      }}>
        {items.length === 0 ? (
          <div style={{ color: '#64748b', fontSize: '12px' }}>No events yet...</div>
        ) : (
          [...items].reverse().map((evt, i) => {
            const time = new Date(evt.timestamp * 1000)
            const timeStr = time.toLocaleTimeString()
            const color = ACTION_COLORS[evt.action_name] || '#64748b'
            return (
              <div key={i} style={{
                fontSize: '11px',
                fontFamily: 'monospace',
                padding: '3px 6px',
                background: 'rgba(15,23,42,0.5)',
                borderRadius: '4px',
                borderLeft: `3px solid ${color}`,
              }}>
                <span style={{ color: '#64748b' }}>[{timeStr}]</span>{' '}
                <span style={{ color, fontWeight: 600 }}>{evt.action_name.toUpperCase()}</span>{' '}
                <span style={{ color: '#cbd5e1' }}>{evt.message}</span>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
