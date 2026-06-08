# SDN DRL-IDS: Rogue AP & ARP Spoofing Detection

Real-time intrusion detection system for SDN networks using Deep Reinforcement Learning (DQN) to detect and isolate **Rogue Access Points** and **ARP Spoofing** attacks.

This is a prototype SDN lab system using Mininet, Open vSwitch, and Ryu. Rogue
AP behavior is represented through packet/flow indicators such as SSID beacon
count and unknown SSID count, not through a physical Wi-Fi access point.

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
| Language | Python 3.11 |

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

**Prerequisite: Python 3.11** (Ryu 4.34 is incompatible with Python 3.12+)

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch python3.11 python3.11-venv python3.11-dev python3-pip
sudo systemctl start openvswitch-switch

# Python environment (must use python3.11)
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install all Python dependencies first (includes ryu's deps)
pip install -r requirements.txt
pip install cryptography

# Install Ryu (requires patch for setuptools compatibility)
cd /tmp
wget https://files.pythonhosted.org/packages/source/r/ryu/ryu-4.34.tar.gz
tar xzf ryu-4.34.tar.gz
cd ryu-4.34
sed -i 's/_main_module()._orig_get_script_args = easy_install.get_script_args/pass/' ryu/hooks.py
pip install . --no-build-isolation --no-deps
cd ~/NWC303

# Let the venv import Ubuntu's Mininet package without overriding venv packages
python - <<'PY'
import site
from pathlib import Path
site_dir = Path(site.getsitepackages()[0])
(site_dir / "ubuntu-dist-packages.pth").write_text("/usr/lib/python3/dist-packages\n")
PY

# Patch Ryu for newer eventlet packages if ryu-manager fails on ALREADY_HANDLED
python - <<'PY'
from pathlib import Path
p = Path(".venv311/lib/python3.11/site-packages/ryu/app/wsgi.py")
s = p.read_text()
old = "    from eventlet.wsgi import ALREADY_HANDLED\n"
new = (
    "    try:\n"
    "        from eventlet.wsgi import ALREADY_HANDLED\n"
    "    except ImportError:\n"
    "        ALREADY_HANDLED = object()\n"
)
if old in s:
    p.write_text(s.replace(old, new))
PY

# Web frontend
cd web/frontend && npm install && npm run build && cd ../..
```

On Ubuntu 26.04/UTM, prefer `.venv311` over an existing `.venv` if `.venv`
was created with Python 3.9 or another Python version.

### 2. Run

```bash
# Terminal 1: Ryu controller
source .venv311/bin/activate
PYTHONPATH=. ryu-manager env/ryu_controller.py --observe-links

# Terminal 2: Train DRL agent (requires sudo, use venv python)
sudo .venv311/bin/python main.py train --algo custom --episodes 100 --max-steps 100

# Terminal 3: Web dashboard
source .venv311/bin/activate
python main.py web --port 8000
# Open http://localhost:8000

# Monitor with TensorBoard
tensorboard --logdir runs/

# Evaluate trained model
sudo .venv311/bin/python main.py evaluate --checkpoint checkpoints/dqn_final.pt --algo custom

# Generate plots
python main.py plot
```

### 3. Test

```bash
source .venv311/bin/activate
pytest tests/ -q
```

If Mininet prints `sch_htb quantum of class ... is big`, it is a noisy Linux
traffic-control warning and is not fatal.

### 4. Run Experiments for the Research Questions

Run a short smoke experiment first:

```bash
sudo .venv311/bin/python main.py train --algo custom --episodes 8 --max-steps 30
sudo .venv311/bin/python main.py experiment --episodes 2 --max-steps 20
```

For report-quality data, increase the budget:

```bash
sudo .venv311/bin/python main.py train --algo custom --episodes 300 --max-steps 100 --seed 42
sudo .venv311/bin/python main.py experiment --episodes 10 --max-steps 100
```

The root-level research scripts provide the same workflow with reward ablation:

```bash
# Full run. This can take a long time because Mininet steps sleep for polling.
sudo PYTHON_BIN=.venv311/bin/python bash run_full_experiment.sh

# Quick smoke run to verify the workflow before a long run.
sudo EPISODES=1 MAX_STEPS=3 EVAL_EPISODES=1 PYTHON_BIN=.venv311/bin/python bash run_full_experiment.sh
```

Reward ablation configs live in `config/reward_v1.yaml`,
`config/reward_v2_fn_penalty.yaml`, and `config/reward_v3_isolate_boost.yaml`.
The default reward in `config.py` uses the stronger FN penalty version.

Generated outputs:

| File | Use |
|---|---|
| `checkpoints/dqn_final.pt` | Final custom DQN checkpoint |
| `runs/metrics.csv` | Per-episode training reward, loss, epsilon |
| `runs/train_steps.csv` | Per-step action, attack type, reward, target DPID/port |
| `runs/experiment_results.csv` | Baseline vs DRL metrics by scenario |
| `runs/confusion_matrix.png` | Confusion matrix plot if `main.py plot` is run |

Use these files to answer the research questions:

| RQ | Evidence to inspect |
|---|---|
| RQ1: Can DRL distinguish normal vs Rogue AP/ARP spoofing? | `runs/experiment_results.csv`: compare recall, precision, F1, TP/FP/TN/FN for `normal`, `arp`, `rogue`, and `mixed` |
| RQ2: Which reward function is effective? | `runs/metrics.csv` and `runs/train_steps.csv`: reward trend, action distribution, missed attack penalties, block/isolate rewards |
| RQ3: Is performance robust outside simulation? | `runs/realtest_results.csv` from `main.py realtest`, compared with Mininet results in `runs/experiment_results.csv` |

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

## Level-2 Real Testbed for RQ3

Use this mode when you want a small real/VM SDN lab instead of Mininet. The
recommended setup is one Ryu controller, one Open vSwitch/OpenFlow switch, and
2-3 real machines or VMs:

```
Victim VM/laptop  ─┐
Normal VM/laptop  ├── Open vSwitch / OpenFlow switch ─── Ryu controller
Attacker VM       ─┘
```

### 1. Prepare Open vSwitch

On the OVS switch machine:

```bash
sudo apt-get install -y openvswitch-switch
sudo ovs-vsctl add-br br-sdn
sudo ovs-vsctl set bridge br-sdn protocols=OpenFlow13
sudo ovs-vsctl set-controller br-sdn tcp:<CONTROLLER_IP>:6653
sudo ovs-vsctl add-port br-sdn <victim_if>
sudo ovs-vsctl add-port br-sdn <normal_if>
sudo ovs-vsctl add-port br-sdn <attacker_if>
sudo ovs-vsctl show
```

If you use VMs on one host, attach VM tap/vnet interfaces to `br-sdn`. If you
use a physical OpenFlow switch, configure its controller as
`tcp:<CONTROLLER_IP>:6653` and note the datapath ID.

### 2. Start Ryu and Dashboard

On the controller machine:

```bash
ryu-manager env/ryu_controller.py --observe-links
python3 main.py web --ryu-url http://<CONTROLLER_IP>:8080 --dpids 1 --sniff-iface any
```

Use the actual DPID if it is not `1`.

### 3. Run Real-Testbed Inference

Start with dry-run mode. It logs what the agent would do but does not install
DROP/VLAN rules:

```bash
python3 main.py realtest \
  --agent custom \
  --checkpoint checkpoints/dqn_final.pt \
  --ryu-url http://<CONTROLLER_IP>:8080 \
  --dpids 1 \
  --steps 120 \
  --ground-truth unknown
```

After confirming target ports are correct, enable real mitigation:

```bash
python3 main.py realtest \
  --agent custom \
  --checkpoint checkpoints/dqn_final.pt \
  --ryu-url http://<CONTROLLER_IP>:8080 \
  --dpids 1 \
  --steps 120 \
  --ground-truth attack \
  --apply-actions
```

Results are written to `runs/realtest_results.csv` with action, inferred attack
type, target DPID/port, target reason, and whether mitigation was applied.

### 4. RQ3 Measurement

For RQ3, train/evaluate in Mininet first, then run the same checkpoint on this
testbed under:

- normal traffic only
- ARP spoofing from the attacker VM
- Rogue AP indicator traffic if available, or the simulated SSID/beacon feature
  path described in the report
- background traffic such as `iperf3`, DNS/HTTP requests, ping bursts, latency,
  and packet loss

Report the performance drop from Mininet to the real/VM testbed using F1, FPR,
MTTD, MTTI, action distribution, and target correctness. If the lab uses VMs or
OVS instead of physical switches, describe it as a **level-2 real/VM OpenFlow
testbed**, not a production network.
