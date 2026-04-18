import random
from typing import Any

from marioai import core

__all__ = ['BaseAgent']


class BaseAgent(core.Agent):
    """Agent that records every state/action/reward it sees.

    The baseline behaviour is to always walk forward with random ``jump`` and
    ``speed`` bits. Subclasses typically override :meth:`act` to plug in a
    policy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.frames = 0
        self.state: dict[str, Any] | None = None
        self.actions: list[list[int]] = []
        self.states: list[dict[str, Any]] = []
        self.rewards: list[dict[str, float]] = []

    def sense(self, state: dict[str, Any]) -> None:
        self.state = state
        self.states.append(state)

    def act(self) -> list[int]:
        action = [0, 1, 0, random.randint(0, 1), random.randint(0, 1)]
        self.actions.append(action)
        return action

    def give_rewards(self, reward: dict[str, float], cum_reward: float) -> None:
        self.rewards.append(reward)
        super().give_rewards(reward, cum_reward)
