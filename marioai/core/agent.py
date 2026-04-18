from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ['Agent']


class Agent:
    """Base class for an autonomous agent.

    Subclasses override :meth:`act` (and optionally :meth:`sense`,
    :meth:`give_rewards`) to produce behaviour. Attributes populated by
    :meth:`sense` mirror the keys of the state dict produced by
    :class:`marioai.core.Task`.

    Attributes:
      level_scene: 22x22 grid of tiles/entities around Mario.
      on_ground: whether Mario is touching the ground.
      can_jump: whether Mario may jump this frame.
      mario_floats: Mario's (x, y) position.
      enemies_floats: flat list of enemy coordinates.
      episode_over: whether the current episode has ended.
    """

    def __init__(self) -> None:
        self.level_scene: np.ndarray | None = None
        self.on_ground: bool | None = None
        self.can_jump: bool | None = None
        self.mario_floats: tuple[float, float] | None = None
        self.enemies_floats: list[float] | None = None
        self.episode_over: bool = False

    def reset(self) -> None:
        """Start a new episode."""
        self.episode_over = False

    def sense(self, state: dict[str, Any]) -> None:
        """Populate attributes from the task's state dict."""
        self.episode_over = state['episode_over']
        self.can_jump = state['can_jump']
        self.on_ground = state['on_ground']
        self.mario_floats = state['mario_floats']
        self.enemies_floats = state['enemies_floats']
        self.level_scene = state['level_scene']

    def act(self) -> list[int]:
        """Return an action as a list of five ints in ``{0, 1}``."""
        return [0, 0, 0, 0, 0]

    def give_rewards(self, reward: dict[str, float], cum_reward: float) -> None:
        """Notify the agent of the latest reward and its cumulative value."""
