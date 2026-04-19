from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from marioai.core.utils import FitnessResult, Observation
from marioai.gym.pixel_environment import ShapedPixelMarioEnv


class FakeCapture:
    """Minimal stand-in for :class:`GameWindowCapture` used in tests."""

    def __init__(self, frames: list[np.ndarray | None], grayscale: bool = True, resize: tuple[int, int] = (84, 84)) -> None:
        self.grayscale = grayscale
        self.resize = resize
        self._frames = list(frames)
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def capture_frame(self) -> np.ndarray | None:
        if self._frames:
            return self._frames.pop(0)
        return None


def _make_env(capture: FakeCapture) -> ShapedPixelMarioEnv:
    """Build a ShapedPixelMarioEnv with the underlying TCP env mocked out."""
    env = ShapedPixelMarioEnv(capture=capture)
    env._env = MagicMock()
    return env


def _frame(value: int = 50) -> np.ndarray:
    return np.full((84, 84), value, dtype=np.uint8)


def _observation() -> Observation:
    return Observation(may_jump=True, on_ground=True, level_scene=np.zeros((22, 22), dtype=int), mario_floats=(0.0, 0.0))


def _fitness(distance: float = 100.0, status: int = 1, coins: int = 3) -> FitnessResult:
    return FitnessResult(status=status, distance=distance, time_left=42, mario_mode=2, coins=coins)


# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_requires_grayscale(self) -> None:
        capture = FakeCapture([_frame()], grayscale=False)
        with pytest.raises(ValueError, match='grayscale=True'):
            ShapedPixelMarioEnv(capture=capture)

    def test_requires_resize(self) -> None:
        capture = FakeCapture([_frame()], resize=None)
        with pytest.raises(ValueError, match='resize'):
            ShapedPixelMarioEnv(capture=capture)

    def test_starts_capture_on_construction(self) -> None:
        capture = FakeCapture([_frame()])
        _make_env(capture)
        assert capture.start_calls == 1

    def test_observation_space(self) -> None:
        capture = FakeCapture([_frame()], resize=(84, 84))
        env = _make_env(capture)
        assert env.observation_space.shape == (84, 84, 1)
        assert env.observation_space.dtype == np.uint8


# ---------------------------------------------------------------------------
class TestObservation:
    def test_step_returns_captured_frame(self) -> None:
        target = _frame(value=200)
        capture = FakeCapture([_frame(value=10), target])
        env = _make_env(capture)
        env._env.get_sensors.side_effect = [_observation(), _observation()]
        env.reset()
        obs, _, done, _ = env.step(0)
        assert not done
        assert np.array_equal(obs.squeeze(-1), target)

    def test_finished_on_fitness_result(self) -> None:
        capture = FakeCapture([_frame()])  # only one frame; ok because terminal step doesn't call capture
        env = _make_env(capture)
        env._env.get_sensors.side_effect = [_observation(), _fitness()]
        env.reset()
        obs, reward, done, _ = env.step(0)
        assert done is True
        # status=1 → +100; distance went 0→100 → +100; coins 0→3 → +30; total = 230
        assert reward == pytest.approx(230.0)
        assert obs.shape == (84, 84, 1)

    def test_missed_frame_reuses_last(self) -> None:
        first = _frame(value=10)
        capture = FakeCapture([first, None, None])
        env = _make_env(capture)
        env._env.get_sensors.side_effect = [_observation(), _observation(), _observation()]
        env.reset()  # consumes 'first' as initial observation
        obs1, _, _, _ = env.step(0)  # capture returns None → reuse first
        obs2, _, _, _ = env.step(0)  # capture returns None again
        assert np.array_equal(obs1.squeeze(-1), first)
        assert np.array_equal(obs2.squeeze(-1), first)

    def test_logs_warning_after_5_misses(self, caplog) -> None:
        first = _frame(value=10)
        capture = FakeCapture([first] + [None] * 5)
        env = _make_env(capture)
        env._env.get_sensors.side_effect = [_observation()] * 6
        env.reset()
        with caplog.at_level(logging.WARNING, logger='marioai.gym.pixel_environment'):
            for _ in range(5):
                env.step(0)
        assert any('5 consecutive frames missed' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
class TestRewardShaping:
    def test_delta_distance_and_coins(self) -> None:
        capture = FakeCapture([_frame()] * 3)
        env = _make_env(capture)
        # reset returns Observation, then two steps with observations
        env._env.get_sensors.side_effect = [_observation(), _observation(), _observation()]
        env.reset()
        # First step: distance went 0→0 (Observation has no distance), so reward=0
        _, r1, _, _ = env.step(0)
        assert r1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_close_stops_capture(self) -> None:
        capture = FakeCapture([_frame()])
        env = _make_env(capture)
        env.close()
        assert capture.stop_calls == 1

    def test_reset_clears_last_obs(self) -> None:
        capture = FakeCapture([_frame(value=10), _frame(value=20)])
        env = _make_env(capture)
        env._env.get_sensors.side_effect = [_observation(), _observation()]
        env.reset()
        assert env._last_obs is not None  # populated by initial _build_observation
        env._env.get_sensors.side_effect = [_observation()]
        env.reset()
        # _last_obs cleared then re-populated by the new initial frame
        assert env._last_obs is not None
