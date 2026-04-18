"""Helpers to derive features from the 22x22 ``level_scene`` grid.

These were originally duplicated between :mod:`marioai.core.task` and
:mod:`marioai.agents.exploratory_agent`; they live here so both can share a
single implementation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = [
    'DEFAULT_OBJECTS',
    'get_ground',
    'has_role_near',
    'is_near',
]


DEFAULT_OBJECTS: dict[str, list[int]] = {
    'soft': [-11],
    'hard': [20, -10],
    'enemy': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
    'brick': [16, 21],
    'projetil': [25],
}


def is_near(
    level_scene: np.ndarray,
    objects: Iterable[int],
    dist: int,
    player_pos: int = 11,
) -> bool:
    """Whether any cell at horizontal offset ``dist`` (rows near the player) matches ``objects``.

    Checks the three rows immediately above the player at column
    ``player_pos + dist``. Returns True on the first hit.
    """
    objects_set = set(objects)
    y = min(level_scene.shape[1] - 1, player_pos + dist)
    for i in range(1, 4):
        x = max(0, player_pos - i)
        if level_scene[x, y] in objects_set:
            return True
    return False


def has_role_near(
    level_scene: np.ndarray,
    ground_pos: int | None,
    dist: int,
    player_pos: int = 11,
) -> bool:
    """Whether the column at ``player_pos + dist`` is clear below ``ground_pos``."""
    if ground_pos is None:
        return False
    y = min(level_scene.shape[1] - 1, player_pos + dist)
    return bool((level_scene[ground_pos:, y] == 0).all())


def get_ground(
    level_scene: np.ndarray,
    on_ground: bool,
    window_size: int = 4,
    player_pos: int = 11,
) -> int:
    """Locate the row index of the first solid tile below the player."""
    if on_ground:
        return player_pos + 1

    start = player_pos - window_size
    end = player_pos + window_size
    is_ground = level_scene[start:, start:end] == -10
    rows = np.nonzero(is_ground)[0]
    if rows.size:
        return int(rows[0] + start)
    return 10
