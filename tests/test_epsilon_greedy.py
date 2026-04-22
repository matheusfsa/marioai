from __future__ import annotations

import numpy as np
import pytest

from marioai.agents import EpsilonGreedyAgent
from marioai.agents.epsilon_greedy_agent import FORWARD_ACTION, FORWARD_BIAS
from marioai.agents.utils import State


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


def _action_pool():
    return np.array(
        [
            [0, 0, 0, 0, 0],  # do nothing  (idx 0)
            [0, 0, 0, 1, 0],  # jump        (idx 1)
            [0, 1, 0, 0, 0],  # forward     (idx 2)
        ]
    )


def test_init_q_biases_forward_for_unseen_states():
    agent = EpsilonGreedyAgent(n_episodes=1)
    agent._action_pool = _action_pool()
    q = agent._init_q(agent._action_pool)
    forward_idx = next(
        i for i, row in enumerate(agent._action_pool)
        if tuple(int(x) for x in row) == FORWARD_ACTION
    )
    assert q[forward_idx] == pytest.approx(FORWARD_BIAS)
    assert q.argmax() == forward_idx


def test_set_deterministic_forces_greedy():
    agent = EpsilonGreedyAgent(n_episodes=1)
    agent.policy_kind = 'e_greedy'
    agent.epsilon = 0.7
    agent.set_deterministic()
    assert agent.policy_kind == 'greedy'
    assert agent.epsilon == 0.0


def test_undiscounted_return_in_step():
    agent = EpsilonGreedyAgent(n_episodes=1, first_visit=True)
    agent._action_pool = _action_pool()
    s1 = _obs()
    s2 = _obs(enemy_1=True)
    key1 = State(**s1)
    key2 = State(**s2)
    agent._Q[key1] = np.zeros(3)
    agent._Q[key2] = np.zeros(3)
    agent.states = [s1, s2]
    agent.actions_idx = [2, 1]
    # Reward shaping: both steps are non-terminal with mario_floats None → 0
    # Force terminal status on last step so return is deterministic.
    agent.rewards = [{}, {'status': 1, 'distance': 100, 'coins': 0}]
    total = agent._step()
    assert total == pytest.approx(50.0)
    # G_0 = G_1 = 50 (γ=1, undiscounted). Action 2 on s1 and action 1 on s2.
    assert agent._Q[key1][2] == pytest.approx(50.0)
    assert agent._Q[key2][1] == pytest.approx(50.0)
    # Untouched entries stay at 0.
    assert agent._Q[key1][0] == 0
    assert agent._Q[key2][2] == 0


def test_first_visit_skips_duplicates():
    agent = EpsilonGreedyAgent(n_episodes=1, first_visit=True)
    agent._action_pool = _action_pool()
    s = _obs()
    key = State(**s)
    agent._Q[key] = np.zeros(3)
    agent.states = [s, s]
    agent.actions_idx = [2, 2]
    agent.rewards = [{'status': 1, 'distance': 0, 'coins': 0}, {}]
    agent._step()
    assert agent._N[key][2] == 1


def test_save_load_roundtrip(tmp_path):
    agent = EpsilonGreedyAgent(n_episodes=5, epsilon_start=0.9, epsilon_end=0.01, decay_fraction=0.5)
    agent._action_pool = _action_pool()
    s = State(**_obs())
    agent._Q[s] = np.array([0.1, 0.2, 0.3])
    agent._N[s] = np.array([1.0, 2.0, 3.0])

    path = tmp_path / 'eps.pkl'
    agent.save(path)
    loaded = EpsilonGreedyAgent.load(path)

    assert loaded.policy_kind == 'greedy'
    assert loaded.epsilon == 0.0
    assert loaded.n_episodes == 5
    assert loaded.epsilon_start == 0.9
    assert loaded.epsilon_end == 0.01
    assert loaded.decay_fraction == 0.5
    assert s in loaded._Q
    assert np.array_equal(loaded._Q[s], [0.1, 0.2, 0.3])
    assert np.array_equal(loaded._N[s], [1.0, 2.0, 3.0])


def test_compute_reward_terminal_signals():
    agent = EpsilonGreedyAgent(n_episodes=1)
    assert agent.compute_reward({'status': 1, 'distance': 300, 'coins': 3}) == pytest.approx(56.0)
    assert agent.compute_reward({'status': 2, 'distance': 50, 'coins': 1}) == pytest.approx(-18.0)
    agent.mario_floats = None
    assert agent.compute_reward({}) == 0.0
