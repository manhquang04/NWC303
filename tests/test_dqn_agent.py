"""Test agent/dqn_agent.py — basic functionality (no Mininet required)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agent.dqn_agent import DQNAgent, QNetwork, ReplayBuffer, Transition  # noqa: E402
from config import CFG  # noqa: E402


def test_qnetwork_forward_shape():
    net = QNetwork(state_dim=20, num_actions=4)
    x = torch.zeros(1, 20)
    y = net(x)
    assert y.shape == (1, 4)


def test_replay_buffer_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(64):
        buf.push(Transition(
            state=np.zeros(20, dtype=np.float32),
            action=i % 4,
            reward=float(i),
            next_state=np.ones(20, dtype=np.float32),
            done=(i == 63),
        ))
    s, a, r, ns, d = buf.sample(32)
    assert s.shape == (32, 20)
    assert a.shape == (32, 1)
    assert r.shape == (32, 1)


def test_agent_act_returns_valid_action():
    agent = DQNAgent(state_dim=20, num_actions=4, device="cpu")
    state = np.zeros(20, dtype=np.float32)
    a = agent.act(state, greedy=True)
    assert 0 <= a < 4


def test_agent_learn_no_crash_with_few_samples():
    agent = DQNAgent(state_dim=20, num_actions=4, device="cpu")
    loss = agent.learn()
    assert loss == 0.0


def test_agent_save_load_checkpoint(tmp_path):
    path = tmp_path / "dqn.pt"
    agent = DQNAgent(state_dim=20, num_actions=4, device="cpu")
    agent.epsilon = 0.25
    agent.save(path)

    loaded = DQNAgent(state_dim=20, num_actions=4, device="cpu")
    loaded.load(path)

    assert path.exists()
    assert loaded.epsilon == 0.25
