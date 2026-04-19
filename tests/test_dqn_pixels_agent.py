from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from marioai.gym.environment import ACTIONS


def _install_fake_sb3(monkeypatch: pytest.MonkeyPatch, action_idx: int = 7) -> MagicMock:
    """Inject a fake ``stable_baselines3`` module with a mocked ``DQN`` class."""
    fake_dqn = MagicMock()
    fake_dqn.predict.return_value = (np.int64(action_idx), None)
    dqn_class = MagicMock()
    dqn_class.load.return_value = fake_dqn
    fake_sb3 = types.ModuleType('stable_baselines3')
    fake_sb3.DQN = dqn_class
    monkeypatch.setitem(sys.modules, 'stable_baselines3', fake_sb3)
    return fake_dqn


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch):
    _install_fake_sb3(monkeypatch, action_idx=7)
    from marioai.agents.dqn_pixels_agent import DqnPixelsAgent

    return DqnPixelsAgent(model_path='fake.zip')


class TestObserveAndAct:
    def test_act_without_frames_returns_noop(self, agent) -> None:
        assert agent.act() == [0, 0, 0, 0, 0]

    def test_pad_stack_when_few_frames(self, agent) -> None:
        agent.observe_frame(np.full((84, 84), 5, dtype=np.uint8))
        action = agent.act()
        assert action == ACTIONS[7]
        # single frame replicated to fill 4-frame stack
        assert len(agent._frames) == 4

    def test_full_stack(self, agent) -> None:
        for i in range(4):
            agent.observe_frame(np.full((84, 84), i, dtype=np.uint8))
        action = agent.act()
        assert action == ACTIONS[7]
        # model.predict called with shape (4, 84, 84) uint8
        args, kwargs = agent.model.predict.call_args
        stacked = args[0]
        assert stacked.shape == (4, 84, 84)
        assert stacked.dtype == np.uint8
        assert kwargs['deterministic'] is True

    def test_none_frame_keeps_stack(self, agent) -> None:
        agent.observe_frame(np.full((84, 84), 5, dtype=np.uint8))
        agent.observe_frame(None)
        assert len(agent._frames) == 1

    def test_reset_clears_stack(self, agent) -> None:
        agent.observe_frame(np.full((84, 84), 5, dtype=np.uint8))
        agent.reset()
        assert len(agent._frames) == 0


class TestPreprocess:
    def test_passes_through_shape_match(self, agent) -> None:
        frame = np.full((84, 84), 3, dtype=np.uint8)
        out = agent._preprocess(frame)
        assert out is frame  # no copy when already conformant

    def test_rgb_converted_to_grayscale_via_numpy(self, agent, monkeypatch) -> None:
        monkeypatch.setattr(agent, '_cv2', None)
        rgb = np.zeros((84, 84, 3), dtype=np.uint8)
        rgb[..., 1] = 200  # pure green → luma ≈ 0.587*200 ≈ 117
        out = agent._preprocess(rgb)
        assert out.shape == (84, 84)
        assert out.dtype == np.uint8
        assert int(out.mean()) == 117


class TestSetDeterministic:
    def test_idempotent(self, agent) -> None:
        agent.set_deterministic()
        agent.set_deterministic()
        assert agent._deterministic is True
