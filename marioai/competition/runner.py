"""Runner que executa um agente nas 5 fases da competição."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from marioai.core import Runner, Task
from marioai.core.agent import Agent

from .phases import PHASES, PhaseConfig

__all__ = ['CompetitionRunner', 'DeterministicAgent', 'PhaseResult']


@runtime_checkable
class DeterministicAgent(Protocol):
    """Interface opcional: agentes estocásticos devem implementar para a avaliação."""

    def set_deterministic(self) -> None: ...


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    status: int
    distance: float
    time_left: int
    coins: int
    mario_mode: int
    wallclock_s: float

    @property
    def won(self) -> bool:
        return self.status == 1


class CompetitionRunner:
    """Roda um agente nas 5 fases, reaproveitando um único :class:`Task`.

    O servidor Java é mantido vivo ao longo das fases (apenas ``env.reset()``
    é chamado entre elas), o que evita o custo de spawn/handshake repetido.
    """

    def __init__(
        self,
        agent: Agent,
        phases: list[PhaseConfig] | None = None,
        max_fps: int = 720,
        visualization: bool = False,
    ) -> None:
        self.agent = agent
        self.phases = phases if phases is not None else PHASES
        self.max_fps = max_fps
        self.visualization = visualization

    def evaluate(self, task: Task | None = None) -> list[PhaseResult]:
        """Executa um episódio por fase; retorna resultados na ordem das fases."""
        if isinstance(self.agent, DeterministicAgent):
            self.agent.set_deterministic()

        owns_task = task is None
        task = task if task is not None else Task()
        results: list[PhaseResult] = []
        try:
            for phase in self.phases:
                results.append(self._run_phase(task, phase))
        finally:
            if owns_task:
                task.disconnect()
        return results

    def _run_phase(self, task: Task, phase: PhaseConfig) -> PhaseResult:
        runner = Runner(
            self.agent,
            task,
            max_fps=self.max_fps,
            level_difficulty=phase.level_difficulty,
            level_type=phase.level_type,
            level_seed=phase.level_seed,
            mario_mode=phase.mario_mode,
            time_limit=phase.time_limit,
            visualization=self.visualization,
        )
        t0 = time.time()
        runner.run()
        wall = time.time() - t0
        reward = task.reward
        return PhaseResult(
            phase=phase.name,
            status=int(reward.get('status', 0) or 0),
            distance=float(reward.get('distance', 0) or 0),
            time_left=int(reward.get('timeLeft', 0) or 0),
            coins=int(reward.get('coins', 0) or 0),
            mario_mode=int(reward.get('marioMode', 0) or 0),
            wallclock_s=round(wall, 3),
        )
