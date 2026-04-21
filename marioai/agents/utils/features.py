from __future__ import annotations

from typing import Any

__all__ = ['TABULAR_STATE_KEYS', 'build_tabular_state']

TABULAR_STATE_KEYS: tuple[str, ...] = (
    'enemy_1',
    'enemy_2',
    'hard_1',
    'soft_1',
    'has_role_near_1',
    'has_role_near_2',
    'can_jump',
    'on_ground',
)


def build_tabular_state(obs: dict[str, Any]) -> tuple[bool, ...]:
    """Compact 8-boolean discretization of ``Task.build_state`` output.

    Shared by all tabular agents (ε-greedy, MC, SARSA, Q-learning) so that
    Q-tables remain comparable across algorithms.
    """
    return tuple(bool(obs.get(k)) for k in TABULAR_STATE_KEYS)
