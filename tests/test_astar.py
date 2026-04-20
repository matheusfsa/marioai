import numpy as np

from marioai.agents.astar_agent import (
    BACKWARD,
    FORWARD,
    FORWARD_JUMP_SPEED,
    GOAL_COL,
    PLAYER_POS,
    AStarAgent,
    path_to_action,
    plan,
)


def _empty_scene() -> np.ndarray:
    return np.zeros((22, 22), dtype=int)


def test_plan_straight_line_on_empty_scene() -> None:
    scene = _empty_scene()
    path = plan(scene, (PLAYER_POS, PLAYER_POS), (PLAYER_POS, GOAL_COL), can_jump=True)
    assert path[0] == (PLAYER_POS, PLAYER_POS)
    assert path[-1] == (PLAYER_POS, GOAL_COL)
    # columns must be non-decreasing (forward motion, possibly with jump arcs)
    cols = [c for _, c in path]
    assert cols == sorted(cols)


def test_plan_returns_empty_when_fully_blocked() -> None:
    scene = _empty_scene()
    # wall of hard tiles everywhere to the right of Mario
    scene[:, PLAYER_POS + 1 :] = -10
    path = plan(scene, (PLAYER_POS, PLAYER_POS), (PLAYER_POS, GOAL_COL), can_jump=True)
    assert path == []


def test_plan_routes_around_obstacle() -> None:
    scene = _empty_scene()
    # vertical wall with one gap above, forcing an up-and-over detour
    scene[PLAYER_POS - 2 :, PLAYER_POS + 2] = -10  # noqa: E203
    path = plan(scene, (PLAYER_POS, PLAYER_POS), (PLAYER_POS, GOAL_COL), can_jump=True)
    assert path, 'expected a path around the obstacle'
    # path must not traverse the wall column at any row >= PLAYER_POS - 2
    for r, c in path:
        if c == PLAYER_POS + 2:
            assert r < PLAYER_POS - 2


def test_path_to_action_forward() -> None:
    assert path_to_action((11, 11), (11, 12), can_jump=True) == FORWARD


def test_path_to_action_backward() -> None:
    assert path_to_action((11, 11), (11, 10), can_jump=True) == BACKWARD


def test_path_to_action_jump_forward() -> None:
    assert path_to_action((11, 11), (10, 12), can_jump=True) == FORWARD_JUMP_SPEED


def test_astar_agent_act_on_empty_scene_returns_forward() -> None:
    agent = AStarAgent()
    agent.sense(
        {
            'level_scene': _empty_scene(),
            'on_ground': True,
            'can_jump': True,
            'mario_floats': (0.0, 0.0),
            'enemies_floats': [],
            'episode_over': False,
        }
    )
    action = agent.act()
    assert action == FORWARD


def test_astar_agent_without_scene_returns_forward() -> None:
    agent = AStarAgent()
    assert agent.act() == FORWARD


def test_astar_agent_reset_clears_plan() -> None:
    agent = AStarAgent()
    agent.sense(
        {
            'level_scene': _empty_scene(),
            'on_ground': True,
            'can_jump': True,
            'mario_floats': (0.0, 0.0),
            'enemies_floats': [],
            'episode_over': False,
        }
    )
    agent.act()
    assert agent._plan_path
    agent.reset()
    assert agent._plan_path == []
    assert agent._plan_index == 0
