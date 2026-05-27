"""Custom DQN with PyTorch (QNetwork, ReplayBuffer, DQNAgent)."""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG

log = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """MLP Q-network: state -> Q(s, .) for num_actions actions."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_layers: Tuple[int, ...] = CFG.dqn.hidden_layers,
    ) -> None:
        super().__init__()
        dims = [state_dim, *hidden_layers]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniform replay buffer."""

    def __init__(self, capacity: int = CFG.dqn.buffer_size) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        s = torch.from_numpy(np.stack([b.state for b in batch])).float()
        a = torch.tensor([b.action for b in batch], dtype=torch.long).unsqueeze(1)
        r = torch.tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(1)
        ns = torch.from_numpy(np.stack([b.next_state for b in batch])).float()
        d = torch.tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(1)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(
        self,
        state_dim: int = CFG.detection.state_dim,
        num_actions: int = 4,
        device: str = CFG.dqn.device,
    ) -> None:
        self.cfg = CFG.dqn
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)

        self.num_actions = num_actions
        self.epsilon = self.cfg.eps_start
        self.steps_done = 0
        self.loss_fn = nn.SmoothL1Loss()

        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(Transition(state, action, reward, next_state, done))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)

    def learn(self) -> float:
        """One Q-network update step. Returns loss (0.0 if insufficient data)."""
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return 0.0

        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)
        s, a, r, ns, d = (x.to(self.device) for x in (s, a, r, ns, d))

        q_sa = self.q_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1, keepdim=True).values
            target = r + self.cfg.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            },
            path,
        )
        log.info("DQN checkpoint saved -> %s", path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.cfg.eps_end)
        self.steps_done = ckpt.get("steps_done", 0)
        log.info("DQN checkpoint loaded <- %s", path)
