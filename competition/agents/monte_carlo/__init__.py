"""Monte Carlo tabular agent for the competition.

Entry points:

- :func:`load_agent` — load the trained Q-table from ``mc_agent.pkl``
  and return a :class:`MonteCarloAgent` in greedy mode, ready for
  :class:`~marioai.competition.CompetitionRunner`.
"""

from __future__ import annotations

from pathlib import Path

from marioai.agents import MonteCarloAgent

__all__ = ['ARTIFACT_PATH', 'load_agent']

ARTIFACT_PATH = Path(__file__).with_name('mc_agent.pkl')


def load_agent(path: str | Path = ARTIFACT_PATH) -> MonteCarloAgent:
    return MonteCarloAgent.load(path)
