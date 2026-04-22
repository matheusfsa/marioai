from __future__ import annotations

import numpy as np
import pytest

from marioai.agents import QLearningAgent
from marioai.agents.q_learning_agent import FORWARD_ACTION, FORWARD_BIAS
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
            [0, 0, 0, 0, 0],  # do nothing
            [0, 0, 0, 1, 0],  # jump
            [0, 1, 0, 0, 0],  # forward
        ]
    )


def _make_agent(**kwargs):
    agent = QLearningAgent(**kwargs)
    agent._action_pool = _action_pool()
    return agent


def test_init_q_biases_forward():
    agent = _make_agent()
    q = agent._init_q(agent._action_pool)
    forward_idx = next(
        i for i, row in enumerate(agent._action_pool)
        if tuple(int(x) for x in row) == FORWARD_ACTION
    )
    assert q[forward_idx] == pytest.approx(FORWARD_BIAS)
    assert q.argmax() == forward_idx


def test_set_deterministic_forces_greedy():
    agent = _make_agent()
    agent.policy_kind = 'e_greedy'
    agent.epsilon = 0.5
    agent.set_deterministic()
    assert agent.policy_kind == 'greedy'
    assert agent.epsilon == 0.0


def test_td_update_non_terminal_uses_max():
    agent = _make_agent(alpha=0.5, gamma=0.9)
    s = State(**_obs())
    s_prime = State(**_obs(enemy_1=True))
    agent._Q[s] = np.array([0.0, 0.0, 0.0])
    agent._Q[s_prime] = np.array([1.0, 5.0, 2.0])  # max = 5
    agent._prev_state = s
    agent._prev_action = 2
    agent._prev_reward = 1.0
    agent._td_update(next_state=s_prime, terminal=False)
    # target = 1 + 0.9 * 5 = 5.5; Q += 0.5 * (5.5 - 0) = 2.75
    assert agent._Q[s][2] == pytest.approx(2.75)
    assert agent._Q[s][0] == 0.0


def test_td_update_terminal_ignores_bootstrap():
    agent = _make_agent(alpha=0.5, gamma=0.9)
    s = State(**_obs())
    agent._Q[s] = np.array([0.0, 0.0, 0.0])
    agent._prev_state = s
    agent._prev_action = 1
    agent._prev_reward = 100.0
    agent._td_update(next_state=None, terminal=True)
    # target = 100; Q += 0.5 * 100 = 50
    assert agent._Q[s][1] == pytest.approx(50.0)


def test_td_update_noop_without_pending_transition():
    agent = _make_agent()
    # Should not raise even though _prev_* are all None.
    agent._td_update(next_state=None, terminal=True)


def test_give_rewards_outside_fit_does_not_update():
    agent = _make_agent()
    s = State(**_obs())
    agent._Q[s] = np.array([0.0, 0.0, 0.0])
    agent._prev_state = s
    agent._prev_action = 1
    agent.in_fit = False
    agent.give_rewards({'status': 1, 'distance': 100, 'coins': 0}, 0.0)
    assert agent._Q[s][1] == 0.0


def test_give_rewards_terminal_records_episode_and_resets():
    agent = _make_agent(alpha=0.5)
    s = State(**_obs())
    agent._Q[s] = np.array([0.0, 0.0, 0.0])
    agent._prev_state = s
    agent._prev_action = 2
    agent.in_fit = True
    agent._episode_reward = 0.0
    agent.give_rewards({'status': 1, 'distance': 100, 'coins': 0}, 0.0)
    # Terminal reward = 100 (status=1, no coins); Q[s][2] += 0.5 * 100
    assert agent._Q[s][2] == pytest.approx(50.0)
    assert agent.fit_rewards[-1] == pytest.approx(100.0)
    assert agent._prev_state is None
    assert agent._prev_action is None


def test_compute_reward_terminal_signals():
    agent = _make_agent()
    assert agent.compute_reward({'status': 1, 'distance': 300, 'coins': 3}) == pytest.approx(130.0)
    assert agent.compute_reward({'status': 2, 'distance': 50, 'coins': 1}) == pytest.approx(-40.0)
    agent.mario_floats = None
    assert agent.compute_reward({}) == 0.0


def test_save_load_roundtrip(tmp_path):
    agent = QLearningAgent(n_episodes=5, alpha=0.2, gamma=0.8, epsilon_start=0.9, epsilon_end=0.02, decay_fraction=0.5)
    agent._action_pool = _action_pool()
    s = State(**_obs())
    agent._Q[s] = np.array([0.1, 0.2, 0.3])

    path = tmp_path / 'ql.pkl'
    agent.save(path)
    loaded = QLearningAgent.load(path)

    assert loaded.policy_kind == 'greedy'
    assert loaded.epsilon == 0.0
    assert loaded.n_episodes == 5
    assert loaded.alpha == 0.2
    assert loaded.gamma == 0.8
    assert loaded.epsilon_start == 0.9
    assert loaded.epsilon_end == 0.02
    assert loaded.decay_fraction == 0.5
    assert s in loaded._Q
    assert np.array_equal(loaded._Q[s], [0.1, 0.2, 0.3])
