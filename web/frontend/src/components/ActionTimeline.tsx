import type { ActionHistoryEntry } from '../types'

const ACTION_COLORS: Record<string, string> = {
  allow: '#22c55e',
  flag: '#eab308',
  block: '#ef4444',
  isolate: '#a855f7',
}

interface Props {
  history: ActionHistoryEntry[]
}

export default function ActionTimeline({ history }: Props) {
  const recent = history.slice(-80)

  return (
    <div style={{
      background: '#1e293b',
      borderRadius: '8px',
      border: '1px solid #334155',
      padding: '16px',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: '#38bdf8' }}>
        Action Timeline
      </div>
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'flex-end',
        gap: '2px',
        overflow: 'hidden',
      }}>
        {recent.map((entry, i) => {
          const color = ACTION_COLORS[entry.action_name] || '#475569'
          return (
            <div
              key={i}
              title={`Step ${entry.step}\nAction: ${entry.action_name}\nGT: ${entry.ground_truth}\nReward: ${(entry.reward ?? 0).toFixed(1)}`}
              style={{
                flex: 1,
                minWidth: '4px',
                maxWidth: '12px',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-end',
              }}
            >
              <div style={{
                height: entry.action === 0 ? '30%' : entry.action === 1 ? '50%' : '80%',
                background: color,
                borderRadius: '2px 2px 0 0',
                transition: 'height 0.2s ease',
                opacity: entry.ground_truth === 'attack' ? 1 : 0.7,
              }} />
            </div>
          )
        })}
      </div>
      {/* Labels */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginTop: '6px',
        fontSize: '10px',
        color: '#64748b',
      }}>
        <span>← older</span>
        <div style={{ display: 'flex', gap: '10px' }}>
          {Object.entries(ACTION_COLORS).map(([name, color]) => (
            <span key={name}>
              <span style={{ color }}>■</span> {name}
            </span>
          ))}
        </div>
        <span>newest →</span>
      </div>
    </div>
  )
}
