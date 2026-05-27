"""Centralized configuration for SDN-DRL-IDS. Import: ``from config import CFG``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT: Path = Path(__file__).parent.resolve()
LOG_DIR: Path = PROJECT_ROOT / "runs"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
DATA_DIR: Path = PROJECT_ROOT / "data"

for _d in (LOG_DIR, CHECKPOINT_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TopologyConfig:
    num_switches: int = 3
    num_hosts: int = 6
    rogue_host_idx: int = 5
    spoofer_host_idx: int = 6
    link_bw_mbps: int = 100
    link_delay: str = "1ms"
    link_loss_pct: float = 0.0


@dataclass(frozen=True)
class RyuConfig:
    controller_ip: str = "127.0.0.1"
    openflow_port: int = 6653
    rest_api_port: int = 8080
    rest_base_url: str = "http://127.0.0.1:8080"
    request_timeout_sec: float = 2.0
    max_retries: int = 3


@dataclass(frozen=True)
class AttackConfig:
    rogue_ssid: str = "FreeWiFi-Evil"
    legit_ssids_whitelist: Tuple[str, ...] = ("CorpNet", "GuestNet")
    rogue_beacon_rate_pps: int = 10
    arp_spoof_rate_pps: int = 50
    arp_spoof_target_ip: str = "10.0.0.1"
    attack_duration_sec_min: int = 5
    attack_duration_sec_max: int = 30


@dataclass(frozen=True)
class DetectionConfig:
    poll_interval_ms: int = 500
    feature_window_sec: float = 1.0
    state_dim: int = 20
    ssid_whitelist: Tuple[str, ...] = ("CorpNet", "GuestNet")
    arp_rate_warn_threshold: float = 20.0
    new_mac_rate_warn_threshold: float = 5.0


@dataclass(frozen=True)
class DQNConfig:
    """DQN hyperparameters (shared by custom PyTorch and SB3)."""

    hidden_layers: Tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    target_update_freq: int = 1_000
    train_freq: int = 4

    # epsilon-greedy exploration (decay per env step)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.9995

    max_episodes: int = 1_000
    max_steps_per_episode: int = 500

    seed: int = 42
    device: str = "auto"
    checkpoint_every: int = 50


ACTION_ALLOW: int = 0
ACTION_FLAG: int = 1
ACTION_BLOCK: int = 2
ACTION_ISOLATE: int = 3
ACTION_NAMES: Tuple[str, ...] = ("allow", "flag", "block", "isolate")
NUM_ACTIONS: int = 4


@dataclass(frozen=True)
class RewardConfig:
    r_attack_blocked: float = +10.0
    r_attack_flagged: float = +2.0
    r_attack_ignored: float = -8.0
    r_normal_allowed: float = +1.0
    r_normal_flagged: float = -0.5
    r_normal_blocked: float = -5.0
    r_time_step: float = -0.1


@dataclass(frozen=True)
class IsolationConfig:
    quarantine_vlan_id: int = 999
    drop_rule_priority: int = 100
    vlan_rule_priority: int = 200
    rule_idle_timeout_sec: int = 60
    rule_hard_timeout_sec: int = 300
    rate_limit_kbps: int = 64


@dataclass(frozen=True)
class LoggingConfig:
    log_level: str = "INFO"
    log_format: str = "[%(asctime)s] %(name)s [%(levelname)s] %(message)s"
    tensorboard_dir: Path = LOG_DIR / "tb"
    csv_metrics_path: Path = LOG_DIR / "metrics.csv"
    confusion_matrix_path: Path = LOG_DIR / "confusion_matrix.png"
    reward_curve_path: Path = LOG_DIR / "reward_curve.png"


@dataclass(frozen=True)
class GlobalConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    ryu: RyuConfig = field(default_factory=RyuConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    logging_cfg: LoggingConfig = field(default_factory=LoggingConfig)


CFG: GlobalConfig = GlobalConfig()
