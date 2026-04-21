from __future__ import annotations

import numpy as np
import pytest

from marioai.agents import MonteCarloAgent
from marioai.agents.utils import State, build_tabular_state, decay_epsilon


def _obs(**overrides):
    base = {
        'enemy_1': False,
        'enemy_2': False,
        'hard_1': False,
        'soft_1': False,
        'has_role_near_1': False,
        'has_role_near_2': False,
        'can_jump': True,
        'on_ground': True,
    }
    base.update(overrides)
    return base


def test_build_tabular_state_returns_eight_bools():
    assert build_tabular_state(_obs()) == (False,) * 6 + (True, True)
    assert build_tabular_state(_obs(enemy_1=1, can_jump=0)) == (
        True, False, False, False, False, False, False, True,
    )


def test_decay_epsilon_linear_endpoints():
    assert decay_epsilon('linear', 0, 10, start=1.0, end=0.1) == pytest.approx(1.0)
    assert decay_epsilon('linear', 9, 10, start=1.0, end=0.1) == pytest.approx(0.1)


def test_decay_epsilon_exponential_endpoints():
    assert decay_epsilon('exponential', 0, 10, start=1.0, end=0.1) == pytest.approx(1.0)
    assert decay_epsilon('exponential', 9, 10, start=1.0, end=0.1) == pytest.approx(0.1)


def test_decay_epsilon_unknown_raises():
    with pytest.raises(ValueError):
        decay_epsilon('bogus', 0, 10)


def test_state_hash_eq_with_numpy():
    a = State(scene=np.array([[1, 2], [3, 4]]), flag=True)
    b = State(scene=np.array([[1, 2], [3, 4]]), flag=True)
    c = State(scene=np.array([[1, 2], [3, 5]]), flag=True)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_monte_carlo_q_update_moves_toward_return():
    agent = MonteCarloAgent(n_samples=1, discount=1.0)
    agent._action_pool = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0]])
    s = _obs()
    state_key = State(**s)
    agent._Q[state_key] = np.zeros(2)
    agent.states = [s]
    agent.actions_idx = [1]
    agent.rewards = [{'distance': 10.0}]
    total = agent._step()
    assert total == pytest.approx(1.0)
    assert agent._Q[state_key][1] > 0
    assert agent._Q[state_key][0] == 0


def test_monte_carlo_save_load_roundtrip(tmp_path):
    agent = MonteCarloAgent(n_samples=3, discount=0.9)
    agent._action_pool = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0]])
    s = State(**_obs())
    agent._Q[s] = np.array([0.5, 1.5])
    agent._N[s] = np.array([2.0, 1.0])

    path = tmp_path / 'mc.pkl'
    agent.save(path)
    loaded = MonteCarloAgent.load(path)

    assert loaded.policy_kind == 'greedy'
    assert loaded.discount == 0.9
    assert loaded.n_samples == 3
    assert s in loaded._Q
    assert np.array_equal(loaded._Q[s], [0.5, 1.5])
    assert np.array_equal(loaded._N[s], [2.0, 1.0])
    assert np.array_equal(loaded._action_pool, agent._action_pool)
