import { useState, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import TopologyGraph from './components/TopologyGraph'
import MetricsPanel from './components/MetricsPanel'
import FeatureChart from './components/FeatureChart'
import ActionTimeline from './components/ActionTimeline'
import EventLog from './components/EventLog'
import ControlPanel from './components/ControlPanel'
import type { TopologyData, ActionHistoryEntry } from './types'

const s = {
  container: {
    display: 'flex', flexDirection: 'column', height: '100vh',
    background: '#0f172a', color: '#e2e8f0',
  } as React.CSSProperties,
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 24px', background: '#1e293b', borderBottom: '1px solid #334155',
  } as React.CSSProperties,
  title: { fontSize: '20px', fontWeight: 700, color: '#38bdf8' } as React.CSSProperties,
  status: { display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' } as React.CSSProperties,
  body: {
    display: 'grid', gridTemplateColumns: '1fr 320px', gridTemplateRows: '1fr 200px',
    gap: '12px', padding: '12px', flex: 1, overflow: 'hidden',
  } as React.CSSProperties,
  topLeft: { display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' } as React.CSSProperties,
  mainGraph: {
    flex: 1, background: '#1e293b', borderRadius: '8px',
    border: '1px solid #334155', overflow: 'hidden', minHeight: 0,
  } as React.CSSProperties,
  rightPanel: {
    display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'auto', gridRow: '1 / 3',
  } as React.CSSProperties,
  bottomLeft: { overflow: 'hidden' } as React.CSSProperties,
}

function Dot({ connected }: { connected: boolean }) {
  return <div style={{
    width: 8, height: 8, borderRadius: '50%',
    background: connected ? '#22c55e' : '#ef4444',
  }} />
}

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

  return (
    <div style={s.container}>
      <div style={s.header}>
        <div style={s.title}>SDN DRL-IDS Dashboard</div>
        <ControlPanel />
        <div style={s.status}>
          <Dot connected={connected} />
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
          {state && (
            <>
              <span style={{ color: '#64748b' }}>|</span>
              <span>Step: {state.step_count}</span>
              <span style={{ color: '#64748b' }}>|</span>
              <span>Ep: {state.episode}</span>
              <span style={{ color: '#64748b' }}>|</span>
              <span>ε: {state.epsilon.toFixed(3)}</span>
            </>
          )}
        </div>
      </div>

      <div style={s.body}>
        <div style={s.topLeft}>
          <div style={s.mainGraph}>
            {topology && state && <TopologyGraph topology={topology} state={state} />}
          </div>
        </div>
        <div style={s.rightPanel}>
          <MetricsPanel metrics={state?.metrics} />
          <FeatureChart state={state} />
          <EventLog events={state?.recent_events} />
        </div>
        <div style={s.bottomLeft}>
          <ActionTimeline history={history} />
        </div>
      </div>
    </div>
  )
}
