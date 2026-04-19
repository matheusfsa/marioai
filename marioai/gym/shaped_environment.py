"""Reward-shaped variant of :class:`MarioEnv` for RL training.

The base env returns Mario's absolute distance as the reward, which is sparse
and biased towards short episodes. ``ShapedMarioEnv`` returns a per-step delta
plus coin/terminal bonuses — much friendlier to value-based methods like DQN.

The shaping formula is intentionally identical to the one documented in
``competition/agents/dqn/02-modelagem.md`` (Etapa 4.1 of the roadmap), so the
symbolic DQN and the pixel DQN see the same reward signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .environment import MarioEnv

__all__ = ['ShapedMarioEnv']


class ShapedMarioEnv(MarioEnv):
    """:class:`MarioEnv` with Δdistance + Δcoins×10 + ±terminal shaping."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prev_distance = 0.0
        self._prev_coins = 0

    def reset(self) -> np.ndarray:
        self._prev_distance = 0.0
        self._prev_coins = 0
        return super().reset()

    def compute_reward(self, reward_data: dict[str, Any]) -> float:
        distance = float(reward_data.get('distance') or 0.0)
        coins = int(reward_data.get('coins') or 0)
        status = reward_data.get('status')

        reward = distance - self._prev_distance
        reward += (coins - self._prev_coins) * 10.0
        if status == 1:
            reward += 100.0  # flag reached
        elif status == 2:
            reward -= 50.0  # death

        self._prev_distance = distance
        self._prev_coins = coins
        return reward
