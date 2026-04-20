import numpy as np

from marioai.agents.astar_agent import (
    BACKWARD,
    FORWARD,
    FORWARD_JUMP,
    FORWARD_JUMP_SPEED,
    GOAL_COL,
    PLAYER_POS,
    AStarAgent,
    path_to_action,
    plan,
)

GROUND_ROW = PLAYER_POS + 1  # 12


def _flat_scene() -> np.ndarray:
    """Cena com chão sólido contínuo em row 12 e ar em tudo acima."""
    scene = np.zeros((22, 22), dtype=int)
    scene[GROUND_ROW, :] = -10
    scene[GROUND_ROW + 1 :, :] = -10
    return scene


def test_plan_walks_on_flat_ground() -> None:
    scene = _flat_scene()
    path = plan(scene, (PLAYER_POS, PLAYER_POS), (PLAYER_POS, GOAL_COL), can_jump=True)
    assert path, 'expected a path along the flat ground'
    assert path[0] == (PLAYER_POS, PLAYER_POS)
    assert path[-1][1] >= GOAL_COL
    # todas as células do caminho em row 11 (sem precisar pular)
    for r, _ in path:
        assert r == PLAYER_POS


def test_plan_empty_scene_walks_along_bottom_row() -> None:
    """Cena totalmente vazia: o suporte "off-window" (fora da janela) é
    considerado sólido, então Mario cai para o row 21 e caminha por ele."""
    scene = np.zeros((22, 22), dtype=int)
    path = plan(scene, can_jump=True)
    assert path, 'expected a fallback path along the bottom row'
    assert path[-1][1] >= GOAL_COL


def test_plan_returns_empty_when_fully_walled() -> None:
    scene = _flat_scene()
    scene[:GROUND_ROW, PLAYER_POS + 1 :] = -10  # noqa: E203
    assert plan(scene, can_jump=True) == []


def test_plan_jumps_over_pit() -> None:
    """Pit de 2 colunas: o plano precisa incluir um salto (dr<0 em algum ponto)."""
    scene = _flat_scene()
    pit_cols = [PLAYER_POS + 3, PLAYER_POS + 4]
    for c in pit_cols:
        scene[GROUND_ROW:, c] = 0  # remove suporte → pit
    path = plan(scene, can_jump=True)
    assert path, 'expected a path jumping over the pit'
    # não pode pousar no fundo do pit (rows abaixo de 11 nas colunas sem suporte)
    for r, c in path:
        if c in pit_cols:
            assert r < GROUND_ROW
    # precisa haver um salto que ultrapasse ≥ 2 colunas (atravessando o pit)
    max_dc = max(abs(path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
    assert max_dc >= 2, 'path must include a jump across the pit columns'


def test_plan_routes_around_wall() -> None:
    scene = _flat_scene()
    # parede de 2 tiles de altura em col 14; gap em cima
    scene[PLAYER_POS, PLAYER_POS + 3] = -10
    scene[PLAYER_POS - 1, PLAYER_POS + 3] = -10
    path = plan(scene, can_jump=True)
    assert path, 'expected a path around the wall'
    # caminho deve ir acima da parede (row < player_pos - 1) na coluna da parede
    for r, c in path:
        if c == PLAYER_POS + 3:
            assert r < PLAYER_POS - 1


def test_path_to_action_forward() -> None:
    assert path_to_action((11, 11), (11, 12), can_jump=True) == FORWARD


def test_path_to_action_backward() -> None:
    assert path_to_action((11, 11), (11, 10), can_jump=True) == BACKWARD


def test_path_to_action_jump_forward() -> None:
    assert path_to_action((11, 11), (10, 12), can_jump=True) == FORWARD_JUMP_SPEED


def test_path_to_action_big_drop_uses_jump() -> None:
    # queda grande à frente — prefere pular para ganhar alcance
    assert path_to_action((11, 11), (13, 12), can_jump=True) == FORWARD_JUMP


def test_astar_agent_act_on_flat_ground_moves_forward() -> None:
    agent = AStarAgent()
    agent.sense(
        {
            'level_scene': _flat_scene(),
            'on_ground': True,
            'can_jump': True,
            'mario_floats': (0.0, 0.0),
            'enemies_floats': [],
            'episode_over': False,
        }
    )
    assert agent.act() == FORWARD


def test_astar_agent_without_scene_returns_forward() -> None:
    agent = AStarAgent()
    assert agent.act() == FORWARD


def test_astar_agent_fallback_forward_jump_when_no_plan() -> None:
    """Parede sólida à frente sem gap → sem plano; agente faz fallback."""
    scene = _flat_scene()
    scene[:GROUND_ROW, PLAYER_POS + 1 :] = -10  # noqa: E203
    agent = AStarAgent()
    agent.sense(
        {
            'level_scene': scene,
            'on_ground': True,
            'can_jump': True,
            'mario_floats': (0.0, 0.0),
            'enemies_floats': [],
            'episode_over': False,
        }
    )
    assert agent.act() == FORWARD_JUMP


def test_astar_agent_reset_clears_plan() -> None:
    agent = AStarAgent()
    agent.sense(
        {
            'level_scene': _flat_scene(),
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
