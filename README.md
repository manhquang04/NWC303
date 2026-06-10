# SDN-IDS: Deep Reinforcement Learning for Intrusion Detection in Software-Defined Networks

> **NWC303 Research Project**

## Abstract

Nghiên cứu này đề xuất hệ thống phát hiện xâm nhập (IDS) dựa trên Deep Q-Network (DQN)
tích hợp với kiến trúc Software-Defined Networking (SDN) sử dụng Ryu controller.
Agent học phân biệt traffic bình thường và tấn công mạng thông qua hàm reward bất đối xứng,
đạt F1 = 0.854 và FPR = 0.126 trên tập kiểm tra UNSW-NB15 (82,332 flows).

## Research Questions

| RQ | Câu hỏi | Kết quả chính |
|---|---|---|
| **RQ1** | DRL agent có thể phân biệt traffic bình thường và tấn công không? | F1=0.854, FPR=0.126 |
| **RQ2** | Thiết kế hàm reward nào hiệu quả nhất? | v3_finetuned: F1=0.898, FPR=0.159 |

## Project Structure

```
NWC303/
├── agent/                    # DQN agent, replay buffer, reward function, Gym wrapper
├── config/                   # Reward ablation YAML configs (v1–v3)
├── config.py                 # Hyperparameters và action definitions
├── dataset/                  # Dataset loaders
│   ├── unsw_nb15_loader.py   # UNSW-NB15 pipeline (primary)
│   └── __init__.py
├── detection/                # Flow collection, feature extraction, state builder
├── env/                      # Mininet topology, Ryu controller, attack simulator, isolator
├── results/                  # Kết quả thực nghiệm (committed)
│   ├── rq1/                  # Per-class metrics, confusion matrix
│   │   ├── per_class_metrics.csv
│   │   ├── confusion_matrix.png
│   │   └── test_metrics.json
│   ├── rq2/                  # Reward ablation comparison, training curves
│   │   ├── ablation_comparison.csv
│   │   ├── training_curves.png
│   │   ├── supplement.txt
│   │   ├── v1_baseline_metrics.json
│   │   ├── v2_balanced_metrics.json
│   │   └── v3_finetuned_metrics.json
│   └── final_report.json
├── train_unsw_nb15.py        # Training entrypoint (UNSW-NB15)
├── evaluate_unsw_nb15.py     # Evaluation entrypoint (UNSW-NB15)
├── train.py                  # Training với Mininet/Ryu (simulation)
├── evaluate.py               # Evaluation với Mininet/Ryu
├── requirements.txt
└── README.md
```

## Dataset

**UNSW-NB15** — [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

| Split | Rows | Nguồn |
|---|---|---|
| Train | 149,039 | Official split |
| Validation | 26,302 | Official split |
| Test | 82,332 | Official split |
| Features | 192 | Sau khi encoding |

> Dataset files không được commit (quá lớn). Tải về và đặt tại `dataset/unsw_nb15/`.

## Installation

Sử dụng Python 3.11. Ryu 4.34 không ổn định trên Python 3.12+.

```bash
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch python3.11 python3.11-venv python3.11-dev python3-pip
sudo systemctl start openvswitch-switch

python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

## Usage

**Train (UNSW-NB15):**
```bash
python3 train_unsw_nb15.py --episodes 100 --max-steps 500 \
  --eval-every 5 --run-dir runs/unsw_nb15_v3
```

**Evaluate:**
```bash
python3 evaluate_unsw_nb15.py --checkpoint runs/unsw_nb15_v3/dqn_best.pt
```

**Reward ablation (RQ2):**
```bash
python3 train_unsw_nb15.py --reward-config v1_baseline --run-dir runs/ablation/v1_baseline
python3 train_unsw_nb15.py --reward-config v2_balanced --run-dir runs/ablation/v2_balanced
python3 train_unsw_nb15.py --reward-config v3_finetuned --run-dir runs/ablation/v3_finetuned
```

**Train với Mininet/Ryu (simulation):**
```bash
# Terminal 1 — Ryu controller
source .venv311/bin/activate
PYTHONPATH=. ryu-manager env/ryu_controller.py --observe-links

# Terminal 2 — Training
sudo .venv311/bin/python train.py \
  --model custom_dqn \
  --episodes 100 \
  --max-steps 100 \
  --attack-ratio 0.4 \
  --save-path runs/checkpoints
```

## Results

### RQ1 — Detection Performance on UNSW-NB15

| Metric | Value |
|---|---|
| Precision | 0.8890 |
| Recall | 0.8218 |
| **F1** | **0.8541** |
| FPR | 0.1257 |
| TP | 37,255 |
| FP | 4,651 |
| TN | 32,349 |
| FN | 8,077 |

Các class khó nhất (rare attacks):
- Worms: F1=0.017 (n=44)
- Shellcode: F1=0.107 (n=378)
- Backdoor: F1=0.137 (n=583)

→ Chi tiết: [`results/rq1/per_class_metrics.csv`](results/rq1/per_class_metrics.csv)
→ Confusion matrix: [`results/rq1/confusion_matrix.png`](results/rq1/confusion_matrix.png)

### RQ2 — Reward Ablation

| Reward Config | Precision | Recall | F1 | FPR | Ep→F1≥0.80 |
|---|---|---|---|---|---|
| v1_baseline (symmetric −3) | 0.460 | 0.095 | 0.157 | 0.136* | Không đạt |
| v2_balanced | 0.866 | 0.932 | 0.898 | 0.176 | Ep 5 |
| **v3_finetuned** | **0.876** | **0.920** | **0.898** | **0.159** | Ep 5 |

*FPR thấp ở v1 do conservative collapse (Recall=0.095), không phải hiệu quả thực sự.

→ Biểu đồ training curves: [`results/rq2/training_curves.png`](results/rq2/training_curves.png)
→ Chi tiết ablation: [`results/rq2/ablation_comparison.csv`](results/rq2/ablation_comparison.csv)

## Action Space

| ID | Action | Hành vi |
|---|---|---|
| 0 | allow | Không can thiệp |
| 1 | flag | Ghi log, tiếp tục giám sát |
| 2 | block | Cài DROP rule trên switch port |
| 3 | isolate | Chuyển port vào quarantine VLAN |

## Reward Design (v3_finetuned — Best Config)

| Tình huống | Reward |
|---|---|
| Allow đúng (normal flow) | +1.5 |
| Flag đúng attack | +2.0 |
| Block đúng attack | +5.0 |
| Isolate đúng attack | +6.0 |
| Bỏ sót attack (missed) | −6.0 |
| Flag nhầm normal | −13.0 |
| Block nhầm normal | −24.0 |
| Isolate nhầm normal | −28.0 |

Tỷ lệ penalty false-positive / missed-attack ≥ 4:1 là điều kiện cần thiết để tránh conservative collapse.

## Reward Configs

Các file cấu hình reward trong `config/`:
- `reward_v1.yaml` — Baseline đối xứng (symmetric −3)
- `reward_v2_balanced.yaml` — Bất đối xứng cơ bản
- `reward_v3_finetuned.yaml` — Bất đối xứng tối ưu (recommended)

## Citation

Nếu sử dụng code hoặc kết quả này, vui lòng trích dẫn:
```
NWC303 SDN-IDS Project, 2026.
Deep Reinforcement Learning for Intrusion Detection in SDN.
Dataset: UNSW-NB15 (Moustafa & Slay, 2015).
```
