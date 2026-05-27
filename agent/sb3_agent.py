"""Stable-Baselines3 DQN wrapper for comparison with custom DQN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from config import CFG

log = logging.getLogger(__name__)


def build_sb3_dqn(env, tensorboard_log: Optional[str] = None) -> DQN:
    cfg = CFG.dqn
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=1.0,                              # hard update via target_update_interval
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        target_update_interval=cfg.target_update_freq,
        exploration_fraction=0.5,
        exploration_initial_eps=cfg.eps_start,
        exploration_final_eps=cfg.eps_end,
        policy_kwargs=dict(net_arch=list(cfg.hidden_layers)),
        seed=cfg.seed,
        device="auto" if cfg.device == "auto" else cfg.device,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model


def train_sb3(model: DQN, total_timesteps: int, save_path: Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cb = CheckpointCallback(
        save_freq=max(1000, CFG.dqn.checkpoint_every * CFG.dqn.max_steps_per_episode),
        save_path=str(save_path.parent),
        name_prefix=save_path.stem,
    )
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    model.save(str(save_path))
    log.info("SB3 model saved → %s", save_path)
