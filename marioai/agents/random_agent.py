import random

from marioai import core

__all__ = ['RandomAgent']


class RandomAgent(core.Agent):
    """Walks forward with random jump/speed bits. Ignores the state."""

    def act(self) -> list[int]:
        return [0, 1, 0, random.randint(0, 1), random.randint(0, 1)]
