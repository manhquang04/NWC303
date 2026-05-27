import { useEffect, useRef } from 'react'
import { Network, DataSet } from 'vis-network/standalone'
import type { TopologyData, DashboardState } from '../types'

const ACTION_COLORS: Record<number, string> = {
  0: '#22c55e', // allow - green
  1: '#eab308', // flag - yellow
  2: '#ef4444', // block - red
  3: '#a855f7', // isolate - purple
}

const ROLE_COLORS: Record<string, string> = {
  normal: '#38bdf8',
  rogue_ap: '#ef4444',
  arp_spoofer: '#f97316',
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
          background: n.type === 'switch' ? '#475569' : (ROLE_COLORS[n.role || 'normal'] || '#38bdf8'),
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
        interaction: {
          hover: true,
          tooltipDelay: 200,
        },
        nodes: {
          borderWidth: 2,
          shadow: true,
        },
        edges: {
          smooth: { enabled: true, type: 'continuous', roundness: 0.5 },
        },
      }
    )
    networkRef.current = network

    return () => {
      network.destroy()
    }
  }, [topology])

  // Update node colors based on action and role
  useEffect(() => {
    if (!nodesRef.current) return

    const isAttack = state.ground_truth === 'attack'
    const updates: any[] = []
    topology.nodes.forEach(n => {
      if (n.type === 'host') {
        let color = ROLE_COLORS[n.role || 'normal'] || '#38bdf8'
        let borderColor = '#64748b'

        if (n.role === 'rogue_ap' || n.role === 'arp_spoofer') {
          if (isAttack && state.current_action > 0) {
            // Attacking host: show action color (flag/block/isolate)
            color = ACTION_COLORS[state.current_action]
            borderColor = '#fbbf24'
          } else if (isAttack) {
            // Attack happening but agent hasn't acted yet
            borderColor = '#fbbf24'
          }
        } else {
          // Normal host: pulse border during attack
          if (isAttack) {
            borderColor = '#f97316'
          }
        }

        updates.push({
          id: n.id,
          color: { background: color, border: borderColor },
        })
      }
    })

    if (updates.length > 0) {
      nodesRef.current.update(updates)
    }
  }, [state.current_action, state.ground_truth, topology.nodes])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: 8,
        left: 8,
        background: 'rgba(15,23,42,0.9)',
        padding: '8px 12px',
        borderRadius: '6px',
        fontSize: '11px',
        display: 'flex',
        gap: '12px',
      }}>
        <span><span style={{ color: '#22c55e' }}>●</span> Allow</span>
        <span><span style={{ color: '#eab308' }}>●</span> Flag</span>
        <span><span style={{ color: '#ef4444' }}>●</span> Block</span>
        <span><span style={{ color: '#a855f7' }}>●</span> Isolate</span>
        <span style={{ color: '#64748b' }}>|</span>
        <span><span style={{ color: '#ef4444' }}>◆</span> Rogue AP</span>
        <span><span style={{ color: '#f97316' }}>◆</span> Spoofer</span>
      </div>
    </div>
  )
}
