"""Validação de Etapa 2: roda AStarAgent nas 5 fases.

Imprime tabela em markdown no stdout.
"""

from __future__ import annotations

import time

from marioai.agents import AStarAgent
from marioai.competition import PHASES, CompetitionRunner


def run_all(label: str, agent_factory, runs_per_phase: int = 1) -> None:
    print(f'\n## {label}\n')
    print('| fase | run | status | distance | time_left | wallclock_s |')
    print('|---|---|---|---|---|---|')
    for phase in PHASES:
        for r in range(runs_per_phase):
            agent = agent_factory()
            runner = CompetitionRunner(agent, phases=[phase], max_fps=720, visualization=False)
            t0 = time.time()
            try:
                results = runner.evaluate()
                res = results[0]
                print(
                    f'| {phase.name} | {r} | {res.status} | {res.distance:.1f} | '
                    f'{res.time_left} | {res.wallclock_s} |'
                )
            except Exception as e:  # noqa: BLE001
                print(f'| {phase.name} | {r} | ERR | {e!r} | - | {time.time() - t0:.1f} |')


if __name__ == '__main__':
    run_all('AStarAgent', AStarAgent, runs_per_phase=1)
