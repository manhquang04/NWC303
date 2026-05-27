import { useEffect, useRef } from 'react'
import { Network, DataSet } from 'vis-network/standalone'
import type { TopologyData, DashboardState } from '../types'

const ACTION_COLORS: Record<number, string> = {
  0: '#22c55e', // allow - green
  1: '#eab308', // flag - yellow
  2: '#ef4444', // block - red
  3: '#a855f7', // isolate - purple
}

const ACTION_LABELS: Record<number, string> = {
  0: 'ALLOW',
  1: 'FLAG',
  2: 'BLOCK',
  3: 'ISOLATE',
}

interface Props {
  topology: TopologyData
  state: DashboardState
}

export default function TopologyGraph({ topology, state }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)
  const nodesRef = useRef<DataSet<any> | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const nodes = new DataSet(
      topology.nodes.map(n => ({
        id: n.id,
        label: n.label,
        shape: n.type === 'switch' ? 'diamond' : 'dot',
        size: n.type === 'switch' ? 30 : 20,
        color: {
          background: n.type === 'switch' ? '#475569' : '#38bdf8',
          border: '#64748b',
          highlight: { background: '#818cf8', border: '#6366f1' },
        },
        font: { color: '#e2e8f0', size: 12 },
        title: n.ip ? `${n.label}\nIP: ${n.ip}\nRole: ${n.role}` : n.label,
        group: n.group,
      }))
    )
    nodesRef.current = nodes

    const edges = new DataSet(
      topology.edges.map((e, i) => ({
        id: i,
        from: e.from,
        to: e.to,
        label: e.label,
        color: { color: '#475569', highlight: '#64748b' },
        font: { color: '#94a3b8', size: 10 },
        width: 2,
      }))
    )

    const network = new Network(
      containerRef.current,
      { nodes, edges },
      {
        physics: {
          enabled: true,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -50,
            springLength: 120,
            springConstant: 0.08,
          },
          stabilization: { iterations: 100 },
        },
        interaction: { hover: true, tooltipDelay: 200 },
        nodes: { borderWidth: 2, shadow: true },
        edges: { smooth: { enabled: true, type: 'continuous', roundness: 0.5 } },
      }
    )
    networkRef.current = network
    return () => { network.destroy() }
  }, [topology])

  // Update node colors based on attack state and action
  useEffect(() => {
    if (!nodesRef.current) return

    const isAttack = state.ground_truth === 'attack'
    const action = state.current_action
    const updates: any[] = []

    topology.nodes.forEach(n => {
      if (n.type !== 'host') return

      const isAttacker = n.role === 'rogue_ap' || n.role === 'arp_spoofer'
      let color = '#38bdf8' // default blue
      let borderColor = '#64748b'
      let label = n.label
      let borderWidth = 2

      if (isAttacker && isAttack) {
        if (action > 0) {
          // Agent acted: show action color
          color = ACTION_COLORS[action]
          borderColor = '#fbbf24'
          label = `${n.label}\n[${ACTION_LABELS[action]}]`
          borderWidth = 4
        } else {
          // Attack happening, agent hasn't acted: red/orange pulse
          color = n.role === 'rogue_ap' ? '#ef4444' : '#f97316'
          borderColor = '#fbbf24'
          label = n.role === 'rogue_ap'
            ? `${n.label}\nROGUE AP`
            : `${n.label}\nSPOOFING`
          borderWidth = 4
        }
      } else if (isAttack && !isAttacker) {
        // Normal host during attack: subtle warning border
        borderColor = '#f97316'
      }

      updates.push({
        id: n.id,
        label,
        color: { background: color, border: borderColor },
        borderWidth,
      })
    })

    if (updates.length > 0) {
      nodesRef.current.update(updates)
    }
  }, [state.current_action, state.ground_truth, topology.nodes])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Attack Alert Banner */}
      {state.ground_truth === 'attack' && (
        <div style={{
          position: 'absolute', top: 8, left: '50%', transform: 'translateX(-50%)',
          background: state.current_action > 0
            ? 'rgba(168,85,247,0.9)' : 'rgba(239,68,68,0.9)',
          padding: '8px 20px', borderRadius: '8px', fontSize: '14px',
          fontWeight: 700, color: '#fff', textAlign: 'center',
          boxShadow: '0 0 20px rgba(239,68,68,0.5)',
          animation: 'pulse 1.5s ease-in-out infinite',
        }}>
          {state.current_action > 0
            ? `ATTACK DETECTED - ${ACTION_LABELS[state.current_action].toUpperCase()}`
            : 'ATTACK IN PROGRESS - MONITORING'}
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute', bottom: 8, left: 8,
        background: 'rgba(15,23,42,0.9)', padding: '8px 12px',
        borderRadius: '6px', fontSize: '11px', display: 'flex', gap: '12px',
      }}>
        <span><span style={{ color: '#22c55e' }}>●</span> Allow</span>
        <span><span style={{ color: '#eab308' }}>●</span> Flag</span>
        <span><span style={{ color: '#ef4444' }}>●</span> Block</span>
        <span><span style={{ color: '#a855f7' }}>●</span> Isolate</span>
        <span style={{ color: '#64748b' }}>|</span>
        <span><span style={{ color: '#ef4444' }}>●</span> Rogue AP</span>
        <span><span style={{ color: '#f97316' }}>●</span> ARP Spoofer</span>
      </div>
    </div>
  )
}
