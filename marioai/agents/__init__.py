from .astar_agent import AStarAgent
from .base_agent import BaseAgent
from .epsilon_greedy_agent import EpsilonGreedyAgent
from .exploratory_agent import ExploratoryAgent
from .monte_carlo_agent import MonteCarloAgent
from .q_learning_agent import QLearningAgent
from .random_agent import RandomAgent

__all__ = [
    'AStarAgent',
    'BaseAgent',
    'DqnPixelsAgent',
    'EpsilonGreedyAgent',
    'ExploratoryAgent',
    'MonteCarloAgent',
    'QLearningAgent',
    'RandomAgent',
]


def __getattr__(name: str):
    """Lazy re-export of :class:`DqnPixelsAgent` to avoid importing SB3 unconditionally."""
    if name == 'DqnPixelsAgent':
        from .dqn_pixels_agent import DqnPixelsAgent

        return DqnPixelsAgent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
