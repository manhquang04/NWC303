import type { DashboardState } from '../types'

const FEATURE_LABELS: Record<string, string> = {
  arp_request_rate: 'ARP Req',
  arp_reply_rate: 'ARP Reply',
  mac_ip_mismatch_count: 'MAC-IP Mis',
  new_mac_rate: 'New MAC',
  ssid_beacon_count: 'SSID Beacon',
  unknown_ssid_count: 'Unknown SSID',
  port_rx_rate_s1: 'Port RX S1',
  port_tx_rate_s1: 'Port TX S1',
  port_rx_rate_s2: 'Port RX S2',
  port_tx_rate_s2: 'Port TX S2',
  flow_count_delta: 'Flow Δ',
  icmp_rate: 'ICMP',
  tcp_syn_rate: 'TCP SYN',
  unique_dst_rate: 'Unique DST',
  packet_size_mean: 'Pkt Size μ',
  packet_size_std: 'Pkt Size σ',
  inter_arrival_mean: 'Inter-arr',
  active_host_count: 'Active Hosts',
  suspicious_port_flag: 'Susp Port',
  time_since_last_alert: 'Since Alert',
}

interface Props { state: DashboardState | null }

export default function FeatureChart({ state }: Props) {
  if (!state || state.state_vector.length === 0) {
    return (
      <div style={{ background: '#111827', borderRadius: '10px', border: '1px solid #1e293b', padding: '14px' }}>
        <div style={{ color: '#4b5563', fontSize: '12px', textAlign: 'center' }}>Waiting for features...</div>
      </div>
    )
  }

  const features = state.state_vector
  const names = state.feature_names || Object.keys(FEATURE_LABELS)

  return (
    <div style={{
      background: '#111827', borderRadius: '10px', border: '1px solid #1e293b',
      padding: '14px', overflow: 'auto', flex: 1,
    }}>
      <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '10px', color: '#38bdf8', letterSpacing: '0.5px' }}>
        STATE VECTOR
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {features.map((val, i) => {
          const name = names[i] || `f${i}`
          const label = FEATURE_LABELS[name] || name
          const suspicious = val > 0.7
          const warning = val > 0.4
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px' }}>
              <span style={{
                width: '70px', color: suspicious ? '#fca5a5' : warning ? '#fde68a' : '#6b7280',
                flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                fontWeight: suspicious ? 700 : 400,
              }}>
                {label}
              </span>
              <div style={{
                flex: 1, height: '10px', background: '#0a0e1a',
                borderRadius: '2px', overflow: 'hidden',
              }}>
                <div style={{
                  width: `${Math.min(val * 100, 100)}%`, height: '100%',
                  background: suspicious
                    ? 'linear-gradient(90deg, #ef4444, #f87171)'
                    : warning
                      ? 'linear-gradient(90deg, #eab308, #fbbf24)'
                      : 'linear-gradient(90deg, #38bdf8, #60a5fa)',
                  borderRadius: '2px',
                  transition: 'width 0.3s ease',
                }} />
              </div>
              <span style={{
                width: '30px', textAlign: 'right', fontFamily: 'monospace',
                color: suspicious ? '#fca5a5' : warning ? '#fde68a' : '#6b7280',
                flexShrink: 0, fontSize: '9px',
              }}>
                {val.toFixed(2)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
