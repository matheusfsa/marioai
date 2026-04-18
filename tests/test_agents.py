import numpy as np
import pytest

from marioai.agents import BaseAgent, ExploratoryAgent, RandomAgent
from marioai.agents.monte_carlo_agent import State


def _mock_state(can_jump: bool = True, on_ground: bool = True) -> dict:
    scene = np.zeros((22, 22), dtype=int)
    return {
        'level_scene': scene,
        'on_ground': on_ground,
        'can_jump': can_jump,
        'mario_floats': (0.0, 0.0),
        'enemies_floats': [],
        'episode_over': False,
    }


class TestRandomAgent:
    def test_act_returns_five_bits(self) -> None:
        agent = RandomAgent()
        for _ in range(10):
            action = agent.act()
            assert len(action) == 5
            assert all(bit in (0, 1) for bit in action)


class TestBaseAgent:
    def test_records_states_and_actions(self) -> None:
        agent = BaseAgent()
        state = _mock_state()
        agent.sense(state)
        agent.act()
        agent.give_rewards({'distance': 1.0}, cum_reward=1.0)
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.rewards) == 1


class TestExploratoryAgent:
    def test_build_state_does_not_mutate_input(self) -> None:
        agent = ExploratoryAgent()
        state = _mock_state()
        snapshot = state['level_scene'].copy()
        agent.sense(state)
        agent.act()
        assert np.array_equal(state['level_scene'], snapshot), 'ExploratoryAgent must not mutate the caller level_scene'


class TestState:
    def test_hash_and_eq_are_consistent(self) -> None:
        a = State(x=1, y=2)
        b = State(x=1, y=2)
        assert a == b
        assert hash(a) == hash(b)

    def test_differs_when_values_differ(self) -> None:
        a = State(x=1, y=2)
        c = State(x=1, y=3)
        assert a != c

    def test_handles_numpy_arrays(self) -> None:
        a = State(grid=np.zeros((3, 3), dtype=int))
        b = State(grid=np.zeros((3, 3), dtype=int))
        assert a == b
        assert hash(a) == hash(b)


class TestAction:
    @pytest.mark.parametrize('agent_cls', [RandomAgent, BaseAgent])
    def test_all_act_outputs_are_valid(self, agent_cls: type) -> None:
        agent = agent_cls()
        for _ in range(20):
            action = agent.act()
            assert isinstance(action, list)
            assert len(action) == 5
            assert all(v in (0, 1) for v in action)
