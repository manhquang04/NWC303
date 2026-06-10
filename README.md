# SDN DRL-IDS: Flow-Based Intrusion Detection with Deep Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple)

## Abstract

This repository implements an SDN-oriented intrusion-detection research prototype using a Deep Q-Network (DQN) agent with four security actions: `allow`, `flag`, `block`, and `isolate`. The current publication pipeline trains and evaluates the agent on the UNSW-NB15 flow dataset, while the original Mininet/Ryu components remain available for SDN lab simulation and enforcement experiments. On the UNSW-NB15 official test split, the best reported DQN policy reaches `Precision=0.8890`, `Recall=0.8218`, `F1=0.8541`, and `FPR=0.1257`. Reward ablation results show that stronger penalties for false intervention on normal flows reduce over-isolation while preserving useful attack detection.

## Research Questions

| RQ | Question | Main Evidence | Result |
|---|---|---|---|
| RQ1 | Can a DRL agent distinguish normal traffic from attack traffic in a flow-based IDS setting? | `results/rq1/test_metrics.json`, `results/rq1/per_class_metrics.csv` | Yes. Overall `F1=0.8541`, `Recall=0.8218`, `FPR=0.1257`. Performance is strongest on high-support classes such as Generic and Exploits, and weakest on rare classes such as Worms and Shellcode. |
| RQ2 | Which reward design is most effective for the DRL-IDS policy? | `results/rq2/ablation_comparison.csv`, `results/rq2/training_curves.png`, `results/rq2/supplement.txt` | `v2_balanced` gives the highest offline F1, while `v3_finetuned` is preferred for deployment because it improves precision and lowers FPR compared with v2. |

## Project Structure

```text
.
├── agent/                         # DQN agent, reward helpers, wrappers
│   ├── dqn_agent.py               # PyTorch DQN implementation
│   ├── reward.py                  # Reward utilities for SDN simulation
│   └── env_wrapper.py             # Action gating / environment wrapper
├── config/                        # Reward YAML files for SDN lab ablations
├── config.py                      # Shared hyperparameters, action IDs, dataset config
├── dataset/
│   ├── unsw_nb15_loader.py        # Main UNSW-NB15 loader
│   └── unsw_nb15/                 # Local raw dataset files, ignored by Git
├── detection/                     # SDN feature extraction and target selection modules
├── env/                           # Mininet/Ryu topology, controller, attack simulator
├── results/
│   ├── rq1/                       # Publication-ready RQ1 metrics and figures
│   ├── rq2/                       # Publication-ready RQ2 ablation evidence
│   └── final_report.json          # Final structured result summary
├── train.py                       # Mininet simulation training entrypoint
├── evaluate.py                    # Mininet simulation evaluation entrypoint
├── train_unsw_nb15.py             # Main UNSW-NB15 DQN training script
├── evaluate_unsw_nb15.py          # Main UNSW-NB15 evaluation script
├── requirements.txt               # Python dependencies
└── README.md
```

`runs/`, `checkpoints/`, raw dataset CSV files, processed caches, and `.pt` checkpoints are local-only and are intentionally ignored by Git.

## Dataset

The current research pipeline uses the UNSW-NB15 dataset:

- Source: [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Mode used here: binary flow classification (`Normal=0`, `Attack=1`)
- Loader: `dataset/unsw_nb15_loader.py`
- Local dataset folder: `dataset/unsw_nb15/`

### Expected Dataset Layout

```text
dataset/unsw_nb15/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
├── UNSW-NB15_4.csv
├── NUSW-NB15_GT.csv
├── NUSW-NB15_features.csv
└── official_split/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

The training pipeline prefers the official `Training and Testing Sets` split. Raw CSV files are not committed because they are large and must be downloaded from the dataset source.

### Split Summary

| Split | Rows | Attack Ratio | Notes |
|---|---:|---:|---|
| Train subset | 149,039 | 68.06% | Derived from official training split after validation split |
| Validation | 26,302 | 68.06% | Stratified validation split |
| Test | 82,332 | 55.06% | Official test split |

## Installation

Use Python 3.11.

```bash
git clone https://github.com/manhquang04/NWC303.git
cd NWC303

python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For the optional Mininet/Ryu SDN simulation pipeline on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch python3.11 python3.11-venv python3.11-dev python3-pip
sudo systemctl start openvswitch-switch
```

Ryu 4.34 may require Python 3.11 and compatibility fixes depending on the host environment. The UNSW-NB15 offline training pipeline does not require Mininet or Ryu.

## Usage

### 1. Train on UNSW-NB15

```bash
source .venv311/bin/activate
python3 train_unsw_nb15.py \
  --episodes 100 \
  --max-steps 500 \
  --eval-every 5 \
  --run-dir runs/unsw_nb15
```

### 2. Evaluate a Trained Checkpoint

```bash
python3 evaluate_unsw_nb15.py \
  --checkpoint runs/unsw_nb15/dqn_best.pt \
  --output-dir runs/unsw_nb15
```

### 3. Reproduce Reward Ablation

The publication-ready ablation outputs are stored under `results/rq2/`. A typical ablation run trains independent agents with different reward settings, then compares `Precision`, `Recall`, `F1`, and `FPR`.

Example configuration style:

```bash
python3 train_unsw_nb15.py \
  --episodes 100 \
  --max-steps 500 \
  --eval-every 5 \
  --run-dir runs/ablation/v3_finetuned \
  --correct-allow 1.5 \
  --attack-flag-reward 2.0 \
  --attack-block-reward 5.0 \
  --attack-isolate-reward 6.0 \
  --missed-attack-penalty 6.0 \
  --normal-flag-penalty 13.0 \
  --normal-block-penalty 24.0 \
  --normal-isolate-penalty 28.0
```

### 4. Optional Mininet/Ryu Simulation

Start the controller:

```bash
PYTHONPATH=. ryu-manager env/ryu_controller.py --observe-links
```

Train in the SDN lab environment:

```bash
sudo .venv311/bin/python train.py \
  --model custom_dqn \
  --episodes 100 \
  --max-steps 100 \
  --attack-ratio 0.4 \
  --save-path runs/checkpoints
```

Evaluate in the SDN lab environment:

```bash
sudo .venv311/bin/python evaluate.py \
  --model runs/checkpoints/dqn_final.pt \
  --scenario all \
  --episodes 20 \
  --max-steps 100 \
  --output runs/evaluation_results.csv
```

## Results

### RQ1: Detection Performance

| Metric | Value |
|---|---:|
| TP | 37,255 |
| FP | 4,651 |
| TN | 32,349 |
| FN | 8,077 |
| Precision | 0.8890 |
| Recall | 0.8218 |
| F1 | 0.8541 |
| FPR | 0.1257 |

Publication artifacts:

- `results/rq1/test_metrics.json`
- `results/rq1/per_class_metrics.csv`
- `results/rq1/confusion_matrix.png`

### RQ2: Reward Ablation

| Reward Config | Precision | Recall | F1 | FPR | First Episode F1 >= 0.80 |
|---|---:|---:|---:|---:|---:|
| `v1_baseline` | 0.4604 | 0.0948 | 0.1572 | 0.1361 | N/A |
| `v2_balanced` | 0.8664 | 0.9319 | 0.8980 | 0.1761 | 5 |
| `v3_finetuned` | 0.8763 | 0.9201 | 0.8977 | 0.1591 | 5 |

Interpretation:

- `v1_baseline` has low FPR but fails because it misses most attacks (`Recall=0.0948`).
- `v2_balanced` gives the highest F1.
- `v3_finetuned` is the preferred deployment configuration because it reduces FPR compared with v2 while keeping F1 high.

Publication artifacts:

- `results/rq2/ablation_comparison.csv`
- `results/rq2/training_curves.png`
- `results/rq2/supplement.txt`
- `results/rq2/v1_baseline_metrics.json`
- `results/rq2/v2_balanced_metrics.json`
- `results/rq2/v3_finetuned_metrics.json`

## Action Space & Reward Design

### Action Space

| ID | Action | Meaning |
|---:|---|---|
| 0 | `allow` | Treat the flow as benign and do not intervene |
| 1 | `flag` | Mark as suspicious for monitoring |
| 2 | `block` | Block suspicious traffic |
| 3 | `isolate` | Quarantine suspicious traffic |

### Reward Design Used by the UNSW-NB15 Pipeline

| Ground Truth | Action | Reward / Penalty |
|---|---|---:|
| Normal | `allow` | `+1.5` |
| Normal | `flag` | `-13.0` |
| Normal | `block` | `-24.0` |
| Normal | `isolate` | `-28.0` |
| Attack | `allow` | `-6.0` |
| Attack | `flag` | `+2.0` |
| Attack | `block` | `+5.0` |
| Attack | `isolate` | `+6.0` |

This reward design explicitly discourages false intervention on normal flows, which is important for SDN environments where unnecessary blocking or isolation can disrupt legitimate traffic.

## Reproducibility Notes

- Raw datasets are not included in the repository.
- Checkpoints are not included in the repository.
- Publication-ready metrics and figures are included under `results/`.
- Local training outputs are written to `runs/`, which is ignored by Git.
- The reported UNSW-NB15 results use binary labels and the official test split.

## Citation

If you use this repository, cite it as:

```bibtex
@misc{nwc303_sdn_drl_ids,
  title        = {SDN DRL-IDS: Flow-Based Intrusion Detection with Deep Reinforcement Learning},
  author       = {Manh Quang},
  year         = {2026},
  howpublished = {\url{https://github.com/manhquang04/NWC303}},
  note         = {Research prototype for SDN-oriented DRL intrusion detection on UNSW-NB15}
}
```

## License

This project is released under the MIT License.
