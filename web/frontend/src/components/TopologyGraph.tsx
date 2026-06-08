import { useEffect, useRef, useState } from 'react'
import { Network, DataSet } from 'vis-network/standalone'
import type { TopologyData, DashboardState } from '../types'

const ACTION_COLORS: Record<number, string> = {
  0: '#22c55e', 1: '#eab308', 2: '#ef4444', 3: '#a855f7',
}
const ACTION_LABELS: Record<number, string> = {
  0: 'ALLOW', 1: 'FLAG', 2: 'BLOCK', 3: 'ISOLATE',
}

interface Props {
  topology: TopologyData
  state: DashboardState
}

function targetHostId(state: DashboardState, topology: TopologyData): string | null {
  if (!state.target) return null
  const mac = state.target.mac?.toLowerCase()
  const macMatch = mac?.match(/00:00:00:00:00:([0-9a-f]{2})$/)
  if (macMatch) {
    const idx = parseInt(macMatch[1], 16)
    if (topology.nodes.some(n => n.id === `h${idx}`)) return `h${idx}`
  }

  const hostsPerSwitch = Math.max(1, Math.floor(topology.config.num_hosts / topology.config.num_switches))
  const hostIdx = (state.target.dpid - 1) * hostsPerSwitch + state.target.port
  return topology.nodes.some(n => n.id === `h${hostIdx}`) ? `h${hostIdx}` : null
}

export default function TopologyGraph({ topology, state }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)
  const nodesRef = useRef<DataSet<any> | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const nodes = new DataSet(
      topology.nodes.map(n => {
        const isSwitch = n.type === 'switch'
        return {
          id: n.id,
          label: n.label,
          shape: isSwitch ? 'diamond' : 'dot',
          size: isSwitch ? 35 : 22,
          color: {
            background: isSwitch ? '#374151' : '#38bdf8',
            border: isSwitch ? '#6b7280' : '#0ea5e9',
            highlight: { background: '#818cf8', border: '#6366f1' },
          },
          font: {
            color: '#e2e8f0', size: 12, face: 'Inter, sans-serif',
            bold: { color: '#fff' },
          },
          title: n.ip
            ? `${n.label}\nIP: ${n.ip}\nRole: ${n.role}\nMAC: ${n.role === 'rogue_ap' ? 'de:ad:be:ef:00:05' : n.role === 'arp_spoofer' ? 'de:ad:be:ef:00:06' : 'auto'}`
            : `${n.label}\nOpenFlow 1.3`,
          group: n.group,
          borderWidth: 2,
          shadow: { enabled: true, color: 'rgba(0,0,0,0.3)', size: 10 },
        }
      })
    )
    nodesRef.current = nodes

    const edges = new DataSet(
      topology.edges.map((e, i) => ({
        id: i, from: e.from, to: e.to, label: e.label,
        color: { color: '#374151', highlight: '#6b7280' },
        font: { color: '#6b7280', size: 9, strokeWidth: 0 },
        width: 2, smooth: { enabled: true, type: 'continuous', roundness: 0.5 },
      }))
    )

    const network = new Network(
      containerRef.current, { nodes, edges },
      {
        physics: {
          enabled: true, solver: 'forceAtlas2Based',
          forceAtlas2Based: { gravitationalConstant: -60, springLength: 130, springConstant: 0.08 },
          stabilization: { iterations: 100 },
        },
        interaction: { hover: true, tooltipDelay: 100, zoomView: true, dragView: true },
        nodes: { borderWidth: 2 },
        edges: { smooth: { enabled: true, type: 'continuous', roundness: 0.5 } },
      }
    )
    networkRef.current = network

    network.on('hoverNode', (params: any) => {
      const nodeId = params.node
      setHoveredNode(nodeId)
    })
    network.on('blurNode', () => setHoveredNode(null))

    return () => { network.destroy() }
  }, [topology])

  // Update colors based on attack/action state
  useEffect(() => {
    if (!nodesRef.current) return

    const isAttack = state.ground_truth === 'attack'
    const action = state.current_action
    const targetId = targetHostId(state, topology)
    const updates: any[] = []

    topology.nodes.forEach(n => {
      if (n.type !== 'host') return

      const isAttacker = n.role === 'rogue_ap' || n.role === 'arp_spoofer'
      const isTarget = targetId === n.id
      let bg = '#38bdf8'
      let border = '#0ea5e9'
      let borderWidth = 2
      let label = n.label
      let shadow: any = { enabled: true, color: 'rgba(0,0,0,0.3)', size: 10 }

      if (isTarget && action > 0) {
        bg = ACTION_COLORS[action]
        border = '#fbbf24'
        borderWidth = 5
        label = `${n.label}\nTARGET ${ACTION_LABELS[action]}`
        shadow = { enabled: true, color: ACTION_COLORS[action], size: 28 }
      } else if (isAttacker && isAttack) {
        if (action > 0) {
          bg = '#7f1d1d'
          border = '#fbbf24'
          borderWidth = 3
          label = `${n.label}\nSUSPECT`
          shadow = { enabled: true, color: '#ef4444', size: 18 }
        } else {
          // Attack happening, no action yet
          bg = n.role === 'rogue_ap' ? '#dc2626' : '#ea580c'
          border = '#fbbf24'
          borderWidth = 4
          label = n.role === 'rogue_ap'
            ? `${n.label}\nROGUE AP`
            : `${n.label}\nARP SPOOF`
          shadow = { enabled: true, color: bg, size: 25 }
        }
      } else if (isAttack && !isAttacker) {
        // Normal host during attack
        border = '#f97316'
        shadow = { enabled: true, color: 'rgba(249,115,22,0.3)', size: 15 }
      }

      updates.push({
        id: n.id, label,
        color: { background: bg, border },
        borderWidth, shadow,
      })
    })

    // Update switch colors
    topology.nodes.forEach(n => {
      if (n.type !== 'switch') return
      updates.push({
        id: n.id,
        color: {
          background: isAttack ? '#1f2937' : '#374151',
          border: isAttack ? '#ef4444' : '#6b7280',
        },
        shadow: { enabled: isAttack, color: 'rgba(239,68,68,0.2)', size: 15 },
      })
    })

    if (updates.length > 0) nodesRef.current.update(updates)
  }, [state.current_action, state.ground_truth, state.target, topology])

  const isAttack = state.ground_truth === 'attack'

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Attack Alert */}
      {isAttack && (
        <div style={{
          position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)',
          background: state.current_action > 0
            ? 'linear-gradient(135deg, rgba(168,85,247,0.9), rgba(139,92,246,0.9))'
            : 'linear-gradient(135deg, rgba(239,68,68,0.9), rgba(220,38,38,0.9))',
          padding: '8px 24px', borderRadius: '20px', fontSize: '13px',
          fontWeight: 700, color: '#fff', textAlign: 'center',
          boxShadow: state.current_action > 0
            ? '0 0 30px rgba(168,85,247,0.5)' : '0 0 30px rgba(239,68,68,0.5)',
          letterSpacing: '1px',
        }}>
          {state.current_action > 0
            ? `${ACTION_LABELS[state.current_action]} TARGET ${state.target ? `s${state.target.dpid}:p${state.target.port}` : 'NOT FOUND'}`
            : 'ATTACK IN PROGRESS — MONITORING'}
        </div>
      )}

      {/* Node Info Tooltip */}
      {hoveredNode && (
        <div style={{
          position: 'absolute', bottom: 50, right: 10,
          background: 'rgba(15,23,42,0.95)', padding: '10px 14px',
          borderRadius: '8px', fontSize: '11px', lineHeight: 1.6,
          border: '1px solid #334155', minWidth: '160px',
        }}>
          <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: '4px' }}>{hoveredNode}</div>
          {(() => {
            const node = topology.nodes.find(n => n.id === hoveredNode)
            if (!node) return null
            if (node.type === 'switch') {
              return <div style={{ color: '#94a3b8' }}>OpenFlow 1.3 Switch</div>
            }
            return (
              <>
                <div style={{ color: '#94a3b8' }}>IP: {node.ip}</div>
                <div style={{ color: '#94a3b8' }}>Role: {node.role}</div>
                {node.role === 'rogue_ap' && <div style={{ color: '#ef4444' }}>Fake SSID: FreeWiFi-Evil</div>}
                {node.role === 'arp_spoofer' && <div style={{ color: '#f97316' }}>Spoofing: 10.0.0.1</div>}
              </>
            )
          })()}
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute', bottom: 8, left: 8,
        background: 'rgba(15,23,42,0.9)', padding: '6px 10px',
        borderRadius: '6px', fontSize: '10px', display: 'flex', gap: '10px',
        color: '#94a3b8',
      }}>
        <span><span style={{ color: '#22c55e' }}>●</span> Allow</span>
        <span><span style={{ color: '#eab308' }}>●</span> Flag</span>
        <span><span style={{ color: '#ef4444' }}>●</span> Block</span>
        <span><span style={{ color: '#a855f7' }}>●</span> Isolate</span>
        <span style={{ color: '#334155' }}>|</span>
        <span><span style={{ color: '#dc2626' }}>●</span> Rogue AP</span>
        <span><span style={{ color: '#ea580c' }}>●</span> Spoofer</span>
      </div>
    </div>
  )
}
