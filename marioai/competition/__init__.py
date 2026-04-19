"""Infraestrutura compartilhada da competição de agentes MarioAI."""

from .phases import PHASES, PhaseConfig
from .runner import CompetitionRunner, DeterministicAgent, PhaseResult
from .scoreboard import AgentScore, Scoreboard

__all__ = [
    'PHASES',
    'AgentScore',
    'CompetitionRunner',
    'DeterministicAgent',
    'PhaseConfig',
    'PhaseResult',
    'Scoreboard',
]
