from unittest.mock import MagicMock, patch

import numpy as np

from marioai.core.task import Task
from marioai.core.utils import FitnessResult, Observation


def _make_task() -> Task:
    """Instantiate Task without actually spawning the Java subprocess."""
    with patch('marioai.core.task.Environment') as env_cls:
        env_cls.return_value = MagicMock()
        task = Task()
        task.env = env_cls.return_value
    return task


def _fake_observation() -> Observation:
    return Observation(
        may_jump=True,
        on_ground=True,
        level_scene=np.zeros((22, 22), dtype=int),
        mario_floats=(1.0, 2.0),
        enemies_floats=[],
    )


class TestBuildState:
    def test_observation_fills_base_keys(self) -> None:
        task = _make_task()
        state = task.build_state(_fake_observation())
        assert state['episode_over'] is False
        assert state['can_jump'] is True
        assert state['on_ground'] is True
        assert state['mario_floats'] == (1.0, 2.0)
        assert state['enemies_floats'] == []
        assert state['level_scene'].shape == (22, 22)

    def test_observation_fills_proximity_features(self) -> None:
        task = _make_task()
        state = task.build_state(_fake_observation())
        for category in ('soft', 'hard', 'enemy', 'brick', 'projetil'):
            for dist in range(1, task.max_dist + 1):
                assert f'{category}_{dist}' in state

    def test_fitness_result_marks_episode_over(self) -> None:
        task = _make_task()
        fit = FitnessResult(status=1, distance=100.0, time_left=20, mario_mode=2, coins=0)
        state = task.build_state(fit)
        assert state['episode_over'] is True
        assert state['level_scene'] is None
        for category in ('soft', 'hard', 'enemy'):
            assert state[f'{category}_1'] is None


class TestFilterActions:
    def test_removes_jumps_when_cannot_jump(self) -> None:
        task = _make_task()
        task.state = {'can_jump': False}
        filtered = task.filter_actions()
        assert (filtered[:, 3] == 0).all(), 'jump column must be all zeros'

    def test_keeps_all_when_state_none(self) -> None:
        task = _make_task()
        task.state = None
        filtered = task.filter_actions()
        assert filtered.shape == task._action_pool.shape

    def test_keeps_jumps_when_can_jump(self) -> None:
        task = _make_task()
        task.state = {'can_jump': True}
        filtered = task.filter_actions()
        assert filtered.shape == task._action_pool.shape


class TestGetSensors:
    def test_fitness_result_sets_finished(self) -> None:
        task = _make_task()
        fit = FitnessResult(status=2, distance=50.0, time_left=0, mario_mode=0, coins=3)
        task.env.get_sensors.return_value = fit
        task.get_sensors()
        assert task.finished is True
        assert task.status == 2
        assert task.reward['distance'] == 50.0

    def test_observation_keeps_finished_false(self) -> None:
        task = _make_task()
        task.env.get_sensors.return_value = _fake_observation()
        task.get_sensors()
        assert task.finished is False
