export interface DashboardState {
  timestamp: number
  state_vector: number[]
  raw_features: Record<string, number>
  feature_names: string[]
  current_action: number
  current_action_name: string
  ground_truth: string
  step_reward: number
  step_count: number
  episode: number
  epsilon: number
  cumulative_reward: number
  metrics: MetricsReport
  recent_events: EventEntry[]
  arp_table: Record<string, string>
  mac_table: Record<string, any>
}

export interface MetricsReport {
  tp: number
  fp: number
  tn: number
  fn: number
  detection_rate: number
  false_positive_rate: number
  precision: number
  recall: number
  f1: number
  mttd_sec: number
  mtti_sec: number
  cumulative_reward: number
  episodes: number
}

export interface EventEntry {
  timestamp: number
  message: string
  action: number
  action_name: string
}

export interface TopologyNode {
  id: string
  label: string
  type: 'switch' | 'host'
  group: string
  ip?: string
  role?: string
}

export interface TopologyEdge {
  from: string
  to: string
  label: string
}

export interface TopologyData {
  nodes: TopologyNode[]
  edges: TopologyEdge[]
  config: {
    num_switches: number
    num_hosts: number
    rogue_host: string
    spoofer_host: string
    link_bw_mbps: number
  }
}

export interface ActionHistoryEntry {
  timestamp: number
  step: number
  action: number
  action_name: string
  ground_truth: string
  reward: number
}

export interface FeatureInfo {
  name: string
  raw_value: number
  normalized_value: number
  max_cap: number
}
