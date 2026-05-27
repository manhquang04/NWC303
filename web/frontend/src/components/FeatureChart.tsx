import type { DashboardState } from '../types'

const FEATURE_LABELS: Record<string, string> = {
  arp_request_rate: 'ARP Req Rate',
  arp_reply_rate: 'ARP Reply Rate',
  mac_ip_mismatch: 'MAC-IP Mismatch',
  new_mac_rate: 'New MAC Rate',
  ssid_beacon_count: 'SSID Beacon',
  unknown_ssid_count: 'Unknown SSID',
  port_rx_rate_s1: 'Port RX S1',
  port_tx_rate_s1: 'Port TX S1',
  port_rx_rate_s2: 'Port RX S2',
  port_tx_rate_s2: 'Port TX S2',
  flow_count_delta: 'Flow Δ',
  icmp_rate: 'ICMP Rate',
  tcp_syn_rate: 'TCP SYN Rate',
  unique_dst_rate: 'Unique DST',
  packet_size_mean: 'Pkt Size μ',
  packet_size_std: 'Pkt Size σ',
  inter_arrival_mean: 'Inter-arr μ',
  active_host_count: 'Active Hosts',
  suspicious_port: 'Susp. Port',
  time_since_alert: 'Since Alert',
}

interface Props {
  state: DashboardState | null
}

export default function FeatureChart({ state }: Props) {
  if (!state || state.state_vector.length === 0) {
    return (
      <div style={{ background: '#1e293b', borderRadius: '8px', border: '1px solid #334155', padding: '16px' }}>
        <div style={{ color: '#64748b', fontSize: '13px' }}>Waiting for features...</div>
      </div>
    )
  }

  const features = state.state_vector
  const names = state.feature_names || Object.keys(FEATURE_LABELS)

  return (
    <div style={{
      background: '#1e293b',
      borderRadius: '8px',
      border: '1px solid #334155',
      padding: '16px',
      overflow: 'auto',
    }}>
      <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#38bdf8' }}>
        State Vector (20 features)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
        {features.map((val, i) => {
          const name = names[i] || `f${i}`
          const label = FEATURE_LABELS[name] || name
          const suspicious = val > 0.7
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
              <span style={{ width: '90px', color: '#94a3b8', flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {label}
              </span>
              <div style={{
                flex: 1,
                height: '12px',
                background: '#0f172a',
                borderRadius: '3px',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${val * 100}%`,
                  height: '100%',
                  background: suspicious ? '#ef4444' : '#38bdf8',
                  borderRadius: '3px',
                  transition: 'width 0.3s ease',
                }} />
              </div>
              <span style={{
                width: '35px',
                textAlign: 'right',
                fontFamily: 'monospace',
                color: suspicious ? '#fca5a5' : '#94a3b8',
                flexShrink: 0,
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
