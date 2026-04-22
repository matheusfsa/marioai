"""ε-greedy tabular agent for the competition.

Entry points:

- :func:`load_agent` — load the trained Q-table from ``eps_agent.pkl`` and
  return an :class:`EpsilonGreedyAgent` in greedy (deterministic) mode, ready
  for :class:`~marioai.competition.CompetitionRunner`.
"""

from __future__ import annotations

from pathlib import Path

from marioai.agents import EpsilonGreedyAgent

__all__ = ['ARTIFACT_PATH', 'load_agent']

ARTIFACT_PATH = Path(__file__).with_name('eps_agent.pkl')


def load_agent(path: str | Path = ARTIFACT_PATH) -> EpsilonGreedyAgent:
    return EpsilonGreedyAgent.load(path)
