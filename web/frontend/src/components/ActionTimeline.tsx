import type { ActionHistoryEntry } from '../types'

const ACTION_COLORS: Record<string, string> = {
  allow: '#22c55e', flag: '#eab308', block: '#ef4444', isolate: '#a855f7',
}

interface Props { history: ActionHistoryEntry[] }

export default function ActionTimeline({ history }: Props) {
  const recent = history.slice(-100)

  return (
    <div style={{
      background: '#111827', borderRadius: '10px', border: '1px solid #1e293b',
      padding: '14px', height: '100%',
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '8px', color: '#38bdf8', letterSpacing: '0.5px' }}>
        ACTION TIMELINE
      </div>
      <div style={{
        flex: 1, display: 'flex', alignItems: 'flex-end', gap: '1px', overflow: 'hidden',
      }}>
        {recent.map((entry, i) => {
          const color = ACTION_COLORS[entry.action_name] || '#374151'
          const isAttack = entry.ground_truth === 'attack'
          return (
            <div
              key={i}
              title={`Step ${entry.step}\nAction: ${entry.action_name}\nGround Truth: ${entry.ground_truth}\nReward: ${(entry.reward ?? 0).toFixed(1)}`}
              style={{
                flex: 1, minWidth: '3px', maxWidth: '8px', height: '100%',
                display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
              }}
            >
              <div style={{
                height: entry.action === 0 ? '20%' : entry.action === 1 ? '45%' : entry.action === 2 ? '70%' : '90%',
                background: color,
                borderRadius: '1px 1px 0 0',
                opacity: isAttack ? 1 : 0.6,
                boxShadow: isAttack && entry.action > 0 ? `0 0 6px ${color}` : 'none',
              }} />
            </div>
          )
        })}
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', marginTop: '6px',
        fontSize: '9px', color: '#4b5563',
      }}>
        <span>older</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          {Object.entries(ACTION_COLORS).map(([name, color]) => (
            <span key={name}>
              <span style={{ color }}>■</span> {name}
            </span>
          ))}
        </div>
        <span>newest</span>
      </div>
    </div>
  )
}
