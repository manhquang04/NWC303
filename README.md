# SDN DRL-IDS: Rogue AP & ARP Spoofing Detection

Real-time intrusion detection system for SDN networks using Deep Reinforcement Learning (DQN) to detect and isolate **Rogue Access Points** and **ARP Spoofing** attacks.

## Architecture

```
┌───────────────────────────────────────────────┐
│           DRL Agent (agent/)                  │
│      DQN · Q-Network · Replay Buffer         │
└──────────────┬────────────┬───────────────────┘
               │ state s_t  │ action a_t
┌──────────────▼────────────▼───────────────────┐
│      Detection Pipeline (detection/)          │
│  flow_collector → feature_extractor → state   │
└──────────────┬────────────────────────────────┘
               │ REST API / OpenFlow 1.3
┌──────────────▼────────────────────────────────┐
│        SDN Network (env/, isolation/)         │
│    Mininet · Ryu Controller · Attack Sim      │
└───────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| SDN Controller | Ryu (OpenFlow 1.3) |
| Network Emulation | Mininet + Open vSwitch |
| DRL Framework | PyTorch (custom DQN) + Stable-Baselines3 |
| RL Environment | Gymnasium |
| Web Dashboard | FastAPI + React + vis-network |
| Language | Python 3.10+ |

## Project Structure

```
├── config.py                  # Centralized hyperparameters
├── main.py                    # Entry point (train/evaluate/web/topo/plot)
├── requirements.txt
│
├── env/                       # SDN environment
│   ├── topology.py            # Mininet topology (3 switches, 6 hosts)
│   ├── ryu_controller.py      # Ryu app + REST API endpoints
│   └── attack_simulator.py    # Rogue AP + ARP spoof attack scripts
│
├── detection/                 # Feature engineering
│   ├── flow_collector.py      # Poll Ryu REST API (500ms interval)
│   ├── feature_extractor.py   # Extract 20 network features
│   ├── state_builder.py       # Normalize features → [0,1] vector
│   └── baseline.py            # Rule-based detector (comparison)
│
├── agent/                     # DRL agent
│   ├── env_wrapper.py         # Gymnasium env wrapping SDN system
│   ├── dqn_agent.py           # Custom PyTorch DQN
│   ├── sb3_agent.py           # Stable-Baselines3 DQN wrapper
│   ├── reward.py              # Reward function
│   └── train.py               # Training loop
│
├── isolation/                 # Action execution
│   ├── isolator.py            # Action dispatcher → Ryu REST
│   ├── flow_rules.py          # OpenFlow rule templates
│   └── vlan_manager.py        # VLAN quarantine logic
│
├── evaluation/                # Metrics & visualization
│   ├── metrics.py             # TP/FP/F1/MTTD/MTTI calculator
│   ├── logger.py              # TensorBoard + CSV logging
│   └── visualizer.py          # Reward curve, confusion matrix
│
├── web/                       # Web dashboard
│   ├── app.py                 # FastAPI backend (WebSocket + REST)
│   ├── bridge.py              # Bridge: SDN modules ↔ web API
│   └── frontend/              # React + TypeScript (vis-network)
│
└── tests/                     # Unit tests (24 tests)
```

## Quick Start

### 1. Install (Ubuntu VM required for Mininet)

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch python3-pip python3-venv
sudo systemctl start openvswitch-switch

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Web frontend
cd web/frontend && npm install && npm run build && cd ../..
```

### 2. Run

```bash
# Terminal 1: Ryu controller
ryu-manager env/ryu_controller.py --observe-links

# Terminal 2: Train DRL agent (requires sudo)
sudo python3 main.py train --algo custom --episodes 1000

# Terminal 3: Web dashboard
python3 main.py web --port 8000
# Open http://localhost:8000

# Monitor with TensorBoard
tensorboard --logdir runs/

# Evaluate trained model
sudo python3 main.py evaluate --checkpoint checkpoints/dqn_final.pt --algo custom

# Generate plots
python3 main.py plot
```

### 3. Test

```bash
pytest tests/ -v
```

## Action Space

| ID | Action | Behavior |
|---|---|---|
| 0 | `allow` | No intervention |
| 1 | `flag` | Log and continue monitoring |
| 2 | `block` | Install DROP rule on port |
| 3 | `isolate` | Move port to quarantine VLAN |

## State Space (20 dimensions)

Normalized network features extracted every 500ms. See `detection/state_builder.FEATURE_ORDER` for the full feature list.

## Reward Function

| Scenario | Reward |
|---|---|
| Attack detected + blocked | +10 |
| Attack detected + flagged | +2 |
| Attack missed (allow) | -8 |
| Normal traffic allowed | +1 |
| False positive (block normal) | -5 |

## Configuration

All hyperparameters centralized in `config.py`. Edit only there — no hardcoded values in modules.

## Web Dashboard

Run `python3 main.py web` and open `http://localhost:8000`:

- **Topology Graph** — Real-time network visualization with color-coded actions
- **Metrics Panel** — TPR, FPR, F1, MTTD, MTTI
- **Feature Chart** — 20-feature state vector as bar chart
- **Action Timeline** — Historical action sequence
- **Event Log** — Real-time attack/isolation events

Deploy to GCP VM: install dependencies, build frontend, run `python3 main.py web --host 0.0.0.0`.
