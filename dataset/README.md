# Dataset

Folder này chứa dataset phục vụ cho research project SDN DRL-IDS.

## Dataset được sử dụng

### ARP Poisoning & Flood in SDN (Mendeley, 2022)
- **Source**: https://data.mendeley.com/datasets/yxzh9fbvbj/2
- **File hiện tại**: `ARP.csv`
- **Labels**: 0 = Benign, 1 = ARP Poison, 2 = ARP Flood
- **Features**: switch_id, in_port, outport, src_mac_addr(eth),
  src_mac_addr(arp), dst_mac_addr(eth), dst_mac_addr(arp), src_ip(arp),
  dst_ip(arp), op_code(arp), packet_in_count, Protocol, Pkt loss, rtt (avg),
  total_time

### InSDN Dataset 2020 (Kaggle)
- **Source**: https://www.kaggle.com/datasets/muhammadumarjavaid/insdn-dataset-2020
- **File hiện tại**: `InSDN.csv`
- **Labels**: Normal, DDoS, Probe, DoS, BFA, U2R
- **Features**: flow-based numeric features such as ports, protocol, duration,
  packet counts, byte counts, flags, IAT, active/idle statistics

## Cách sử dụng

```bash
python3 dataset/verify_datasets.py
python3 evaluate_real.py --dataset arp --model runs/checkpoints/dqn_final.pt
python3 evaluate_real.py --dataset insdn --model runs/checkpoints/dqn_final.pt
python3 experiment_real.py --dataset arp --episodes 50
```

CSV dataset files are intentionally ignored by Git. Keep the large downloaded
files local under `dataset/`.
