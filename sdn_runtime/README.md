# SDN Runtime Components

This directory contains the runtime components for SDN and live Wi-Fi validation.

## Files

| File | Purpose |
|---|---|
| `arp_guard_controller.py` | Ryu controller with ARP observation, DQN-style policy decision, alerting, and drop-rule mitigation. |
| `run_arp_sdn_runtime.py` | Runs a single Mininet/Ryu ARP runtime episode. |
| `run_arp_sdn_runtime_benchmark.py` | Runs repeated ARP runtime episodes, including stress and hardcore grids. |
| `rogue_ap_live_sensor.py` | Reads monitor-mode 802.11 traffic with tshark, aggregates windows, and applies model/policy decisions. |
| `run_rogue_ap_live_attack_test.py` | Live Rogue AP beacon-injection helper. |
| `rogue_ap_replay_digital_twin.py` | Digital-twin/replay style Rogue AP runtime helper. |
| `setup_wifi_monitor_vm.sh` | Utility for Wi-Fi monitor-mode setup in the VM. |

## Hardcore ARP Runtime

The current hardcore benchmark adds:

- Evasive attacks: `request_poison`, `unicast_reply`, `burst_then_sleep`.
- Benign hard cases: `gateway_failover`, `ip_reassignment`, `arp_storm`.
- Bursty background mode: `burst`.

These hard cases are intended to expose robustness boundaries, not to maximize scores.

