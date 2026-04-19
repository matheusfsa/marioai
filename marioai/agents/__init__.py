from .base_agent import BaseAgent
from .exploratory_agent import ExploratoryAgent
from .monte_carlo_agent import MonteCarloAgent
from .random_agent import RandomAgent

__all__ = ['BaseAgent', 'DqnPixelsAgent', 'ExploratoryAgent', 'MonteCarloAgent', 'RandomAgent']


def __getattr__(name: str):
    """Lazy re-export of :class:`DqnPixelsAgent` to avoid importing SB3 unconditionally."""
    if name == 'DqnPixelsAgent':
        from .dqn_pixels_agent import DqnPixelsAgent

        return DqnPixelsAgent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
