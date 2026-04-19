"""Configuração das 5 fases de avaliação da competição.

Cada ``PhaseConfig`` fixa os parâmetros passados ao servidor Java antes de
``env.reset()``. A tabela abaixo espelha ``competition/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ['PHASES', 'PhaseConfig']


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    level_difficulty: int
    level_type: int
    level_seed: int
    mario_mode: int
    time_limit: int


PHASES: list[PhaseConfig] = [
    PhaseConfig('1-easy', 0, 0, 1001, 2, 60),
    PhaseConfig('2-medium-A', 5, 0, 2042, 2, 80),
    PhaseConfig('3-medium-B', 8, 1, 2077, 2, 100),
    PhaseConfig('4-hard-A', 15, 2, 3013, 1, 120),
    PhaseConfig('5-hard-B', 20, 3, 3099, 0, 120),
]
