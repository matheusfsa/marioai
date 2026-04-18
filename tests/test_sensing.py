import numpy as np

from marioai.core import sensing


def test_is_near_finds_object_above_player() -> None:
    scene = np.zeros((22, 22), dtype=int)
    scene[10, 12] = 2  # goomba one row above and one col right of Mario
    assert sensing.is_near(scene, objects=[2], dist=1) is True


def test_is_near_misses_when_no_match() -> None:
    scene = np.zeros((22, 22), dtype=int)
    assert sensing.is_near(scene, objects=[2, 3], dist=2) is False


def test_has_role_near_true_on_clear_column() -> None:
    scene = np.zeros((22, 22), dtype=int)
    assert sensing.has_role_near(scene, ground_pos=12, dist=1) is True


def test_has_role_near_false_when_column_blocked() -> None:
    scene = np.zeros((22, 22), dtype=int)
    scene[15, 12] = -10
    assert sensing.has_role_near(scene, ground_pos=12, dist=1) is False


def test_has_role_near_false_when_ground_pos_none() -> None:
    scene = np.zeros((22, 22), dtype=int)
    assert sensing.has_role_near(scene, ground_pos=None, dist=1) is False


def test_get_ground_when_on_ground_returns_player_plus_one() -> None:
    scene = np.zeros((22, 22), dtype=int)
    assert sensing.get_ground(scene, on_ground=True) == 12


def test_get_ground_when_airborne_finds_solid_row() -> None:
    scene = np.zeros((22, 22), dtype=int)
    scene[15, 9] = -10
    ground = sensing.get_ground(scene, on_ground=False)
    assert ground == 15
