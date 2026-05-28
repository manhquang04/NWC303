import { useState, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import TopologyGraph from './components/TopologyGraph'
import MetricsPanel from './components/MetricsPanel'
import FeatureChart from './components/FeatureChart'
import ActionTimeline from './components/ActionTimeline'
import EventLog from './components/EventLog'
import ControlPanel from './components/ControlPanel'
import type { TopologyData, ActionHistoryEntry } from './types'

export default function App() {
  const { state, connected } = useWebSocket()
  const [topology, setTopology] = useState<TopologyData | null>(null)
  const [history, setHistory] = useState<ActionHistoryEntry[]>([])

  useEffect(() => {
    fetch('/api/topology').then(r => r.json()).then(setTopology).catch(() => {})
  }, [])

  useEffect(() => {
    if (state) {
      fetch('/api/history?limit=100').then(r => r.json()).then(d => setHistory(d.history || [])).catch(() => {})
    }
  }, [state?.step_count])

  const isAttack = state?.ground_truth === 'attack'

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100vh',
      background: '#0a0e1a', color: '#e2e8f0',
      fontFamily: "'Inter', -apple-system, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 20px',
        background: isAttack ? 'linear-gradient(90deg, #1a0a0a, #1e293b)' : '#151b2e',
        borderBottom: `2px solid ${isAttack ? '#ef4444' : '#1e293b'}`,
        transition: 'all 0.5s ease',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ fontSize: '18px', fontWeight: 800, color: '#38bdf8', letterSpacing: '-0.5px' }}>
            SDN DRL-IDS
          </div>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? '0 0 8px #22c55e' : '0 0 8px #ef4444',
          }} />
          <span style={{ fontSize: '12px', color: connected ? '#86efac' : '#fca5a5' }}>
            {connected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>

        <ControlPanel />

        <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#94a3b8' }}>
          <span>Step <b style={{ color: '#e2e8f0' }}>{state?.step_count ?? 0}</b></span>
          <span>Episode <b style={{ color: '#e2e8f0' }}>{state?.episode ?? 0}</b></span>
          <span>Epsilon <b style={{ color: '#e2e8f0' }}>{(state?.epsilon ?? 0).toFixed(3)}</b></span>
          {isAttack && (
            <span style={{
              color: '#ef4444', fontWeight: 700,
              animation: 'pulse 1s ease-in-out infinite',
            }}>
              ATTACK DETECTED
            </span>
          )}
        </div>
      </div>

      {/* Main Body */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 300px',
        gridTemplateRows: '1fr 180px',
        gap: '10px', padding: '10px', flex: 1, overflow: 'hidden',
      }}>
        {/* Topology Graph - main area */}
        <div style={{
          background: '#111827', borderRadius: '10px',
          border: `1px solid ${isAttack ? '#7f1d1d' : '#1e293b'}`,
          overflow: 'hidden', minHeight: 0,
          transition: 'border-color 0.5s ease',
        }}>
          {topology && state && <TopologyGraph topology={topology} state={state} />}
        </div>

        {/* Right Panel */}
        <div style={{
          display: 'flex', flexDirection: 'column', gap: '10px',
          overflow: 'auto', gridRow: '1 / 3',
        }}>
          <MetricsPanel metrics={state?.metrics} />
          <FeatureChart state={state} />
          <EventLog events={state?.recent_events} />
        </div>

        {/* Bottom - Action Timeline */}
        <div style={{ overflow: 'hidden' }}>
          <ActionTimeline history={history} />
        </div>
      </div>
    </div>
  )
}
