from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .task import Task

__all__ = ['Experiment']


class Experiment:
    """Runs episodic interaction between a :class:`Task` and an :class:`Agent`."""

    def __init__(
        self,
        task: Task,
        agent: Agent,
        response_delay: int = 0,
    ) -> None:
        self.response_delay = response_delay
        self._frame = 0
        self.task = task
        self.agent = agent
        self.max_fps: int = -1
        self.action: int = 0

    def _step(self) -> dict[str, float]:
        state = self.task.get_sensors()
        if self.task.finished or (self._frame % (self.response_delay + 1)) == 0:
            self.agent.sense(state)
            self.task.perform_action(self.agent.act())
            self.agent.give_rewards(self.task.reward, self.task.cum_reward)
        else:
            self.task.perform_action([0, 0, 0, 0, 0])
        self._frame += 1
        return self.task.reward

    def _episode(self) -> list[dict[str, float]]:
        rewards: list[dict[str, float]] = []
        self.agent.reset()
        self.task.reset()
        while not self.task.finished:
            rewards.append(self._step())
            if self.max_fps > 0:
                time.sleep(1.0 / self.max_fps)
        return rewards

    def do_episodes(self, n: int = 1) -> list[list[dict[str, float]]]:
        return [self._episode() for _ in range(n)]
