# SDN DRL-IDS: Rogue AP & ARP Spoofing Detection

Prototype SDN lab system for detecting Rogue AP indicators and ARP spoofing in
Mininet/Ryu using Deep Reinforcement Learning. Rogue AP behavior is represented
through packet/flow features such as SSID beacon count and unknown SSID count,
not through a physical Wi-Fi access point.

## Core Pipeline

```text
agent/       DQN agent, replay buffer, reward function, Gym wrapper
config/      Reward ablation YAML files
config.py    Central hyperparameters and action definitions
detection/   Flow collection, feature extraction, state builder, baseline
env/         Mininet topology, Ryu controller, attack simulator, isolation logic
dataset/     External dataset notes and manual download instructions
train.py     Training entry point
evaluate.py  Evaluation metrics entry point
experiment.py Reward-ablation runner
evaluate_real.py Real CSV dataset evaluation without Ryu/Mininet
experiment_real.py Real CSV reward-ablation runner without Ryu/Mininet
run_full_experiment.sh Full training/evaluation/ablation workflow
```

Generated outputs are written to `runs/`, `results/`, and `checkpoints/`.

## Install on Ubuntu VM

Use Python 3.11. Ryu 4.34 is not reliable on Python 3.12+.

```bash
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch python3.11 python3.11-venv python3.11-dev python3-pip
sudo systemctl start openvswitch-switch

python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
pip install cryptography
```

Install Ryu with the setuptools compatibility patch:

```bash
cd /tmp
wget https://files.pythonhosted.org/packages/source/r/ryu/ryu-4.34.tar.gz
tar xzf ryu-4.34.tar.gz
cd ryu-4.34
sed -i 's/_main_module()._orig_get_script_args = easy_install.get_script_args/pass/' ryu/hooks.py
pip install . --no-build-isolation --no-deps
cd ~/NWC303
```

Let the venv import Ubuntu's Mininet package:

```bash
python - <<'PY'
import site
from pathlib import Path
site_dir = Path(site.getsitepackages()[0])
(site_dir / "ubuntu-dist-packages.pth").write_text("/usr/lib/python3/dist-packages\n")
PY
```

Patch Ryu if `ryu-manager` fails on `ALREADY_HANDLED`:

```bash
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
```

## Run

Start the Ryu controller in one terminal:

```bash
source .venv311/bin/activate
PYTHONPATH=. ryu-manager env/ryu_controller.py --observe-links
```

Train a custom DQN agent:

```bash
sudo .venv311/bin/python train.py \
  --model custom_dqn \
  --episodes 100 \
  --max-steps 100 \
  --attack-ratio 0.4 \
  --save-path runs/checkpoints
```

Evaluate the trained checkpoint:

```bash
sudo .venv311/bin/python evaluate.py \
  --model runs/checkpoints/dqn_final.pt \
  --scenario all \
  --episodes 20 \
  --max-steps 100 \
  --output runs/evaluation_results.csv
```

Run a reward ablation experiment:

```bash
sudo .venv311/bin/python experiment.py \
  --reward config/reward_v6_logic_fixed.yaml \
  --episodes 50 \
  --max-steps 50 \
  --eval-episodes 10 \
  --attack-ratio 0.4 \
  --output runs/exp_v6.csv
```

Run the full workflow:

```bash
sudo EPISODES=50 MAX_STEPS=50 EVAL_EPISODES=10 ATTACK_RATIO=0.4 \
  PYTHON_BIN=.venv311/bin/python bash run_full_experiment.sh
```

## Outputs for Research Questions

| File | Use |
|---|---|
| `runs/checkpoints/dqn_final.pt` | Final custom DQN checkpoint |
| `runs/checkpoints/reward_curve.png` | Training reward graph |
| `runs/metrics.csv` | Episode reward, loss, epsilon |
| `runs/train_steps.csv` | Step-level action, reward, target, attack type |
| `runs/step_debug.csv` | Debug trace for proposed vs executed actions |
| `runs/evaluation_results.csv` | Custom DQN metrics by scenario |
| `runs/final_research_results.csv` | Reward ablation summary |

Use the files as follows:

| RQ | Evidence |
|---|---|
| RQ1: Can DRL distinguish normal vs Rogue AP/ARP spoofing? | `runs/evaluation_results.csv`: TP, FP, TN, FN, recall, precision, F1 for `normal`, `arp`, `rogue`, `mixed` |
| RQ2: Which reward function is effective? | `runs/final_research_results.csv`: compare reward configs v1-v6 by Normal FPR, attack recall, and F1 |
| RQ3: Is performance robust outside simulation? | Requires a labeled real/VM OpenFlow testbed; Mininet results alone are not enough |

## Dataset

Dataset notes live in `dataset/README.md`. The external ARP dataset is not
committed; download `arp_stats.csv` manually from Mendeley and place it at
`dataset/arp_stats.csv`.

For the macOS-only real dataset workflow:

```bash
python3 dataset/verify_datasets.py
python3 evaluate_real.py --dataset arp --model runs/checkpoints/dqn_final.pt
python3 evaluate_real.py --dataset insdn --model runs/checkpoints/dqn_final.pt
python3 experiment_real.py --dataset arp --episodes 50
python3 experiment_real.py --dataset insdn --episodes 50
```

If a real dataset has a different feature count from the synthetic SDN state,
`evaluate_real.py` pads or truncates features to match the checkpoint input
dimension. `experiment_real.py` trains a dataset-specific DQN input layer.

## Action Space

| ID | Action | Behavior |
|---:|---|---|
| 0 | `allow` | No intervention |
| 1 | `flag` | Log and continue monitoring |
| 2 | `block` | Install DROP rule on suspicious port |
| 3 | `isolate` | Move suspicious port into quarantine VLAN |

## Reward Configs

Reward ablation files are in `config/`:

- `reward_v1.yaml`
- `reward_v2_fn_penalty.yaml`
- `reward_v3_isolate_boost.yaml`
- `reward_v4_balanced.yaml`
- `reward_v5_conservative.yaml`
- `reward_v6_logic_fixed.yaml`

The current pipeline keeps proposed and executed actions separate in logs, and
the replay buffer learns from the executed action selected by the environment.
